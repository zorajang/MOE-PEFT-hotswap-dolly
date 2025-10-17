# moe_peft/adapters/merging.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Any
import re
import torch
import torch.nn as nn


# ====== Public Spec ======

StrategyT = Literal["mean", "task_weighted", "fisher_weighted", "joint_svd"]
ScopeT = Literal["global", "layerwise", "blockwise"]

@dataclass
class MergeSpec:
    strategy: StrategyT = "mean"
    scope: ScopeT = "layerwise"
    experts: List[int] | Literal["all"] = "all"
    alpha: float = 0.5                       # 用于混合/平滑的系数（部分策略可用）
    task_weights: Optional[Dict[int, float]] = None   # strategy == task_weighted
    fisher_stats: Optional[Dict[str, float]] = None   # strategy == fisher_weighted，key 形如 "{module_key}::expert_{i}"
    target_rank: Optional[int] = None                # strategy == joint_svd 时可以下调 rank
    name_suffix: str = "-merged"                     # 导出适配器名后缀（如 "mixlora_0-merged"）
    dtype: Optional[torch.dtype] = None              # 合并时使用的计算精度（默认 fp32）
    device: Optional[torch.device] = None            # 合并时放置的设备（默认 cpu）


# ====== Utilities ======

_LORA_A_KEYS = ("lora_A", "lora_A_", "A", "down", "lora_down")
_LORA_B_KEYS = ("lora_B", "lora_B_", "B", "up", "lora_up")

_EXPERT_NAME_RE = re.compile(r".*?moe\.(?P<moe_name>[^.]+)\.experts\.(?P<idx>\d+)$")


def _tensor_dtype(spec: MergeSpec) -> torch.dtype:
    return spec.dtype or torch.float32


def _to_dtype(x: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    return x.to(dt) if x.dtype != dt else x


def _normalize(weights: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(0.0, float(w)) for w in weights.values()) or 1.0
    return {k: float(max(0.0, w)) / s for k, w in weights.items()}


def _module_key_from_path(path: str) -> str:
    # 统一模块 key，用于报告与统计
    return path


def _get_attr_any(obj: Any, names: Tuple[str, ...]) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _extract_AB_from_lora_module(lora_mod: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    以**尽量鲁棒**的方式获取 LoRA 的 A/B 权重：
    1) 直接读常见属性（lora_A/lora_B 或 lora_down/lora_up 等）；
    2) 否则 fallback 到 state_dict 里查 'lora_A.weight'/'lora_B.weight'。
    """
    A = _get_attr_any(lora_mod, _LORA_A_KEYS)
    B = _get_attr_any(lora_mod, _LORA_B_KEYS)

    if isinstance(A, nn.Module): A = getattr(A, "weight", None)
    if isinstance(B, nn.Module): B = getattr(B, "weight", None)

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return A, B

    sd = lora_mod.state_dict()
    A_key = next((k for k in sd.keys() if k.endswith("lora_A.weight") or k.endswith("lora_down.weight")), None)
    B_key = next((k for k in sd.keys() if k.endswith("lora_B.weight") or k.endswith("lora_up.weight")), None)
    if A_key is None or B_key is None:
        raise RuntimeError(f"Cannot locate LoRA A/B weights in module: {lora_mod.__class__.__name__}")
    return sd[A_key], sd[B_key]


def _put_AB_into_lora_weightdict(weight_dict: Dict[str, torch.Tensor], module_key: str,
                                 adapter_name: str, A: torch.Tensor, B: torch.Tensor):
    """
    产出与现有 init_adapter(..., lora_weight) 兼容的 weight dict 结构。
    这里使用统一 key 命名："{module_key}::{adapter_name}::A/B"
    上层再做具体的适配（例如写回 Linear.loras_[adapter_name] 的内部格式）。
    """
    weight_dict[f"{module_key}::{adapter_name}::A"] = A
    weight_dict[f"{module_key}::{adapter_name}::B"] = B


# ====== Collector ======

class LoRAExpertCollector:
    """
    在整个 model tree 中，查找拥有 `loras_` 字段的 Linear wrapper（或同类），
    并抽取 'moe.{name}.experts.{idx}' 形式的专家权重。
    """
    def __init__(self, model: nn.Module, adapter_name_filter: Optional[str] = None):
        self.model = model
        self.adapter_name_filter = adapter_name_filter

    def iter_lora_wrappers(self):
        for name, mod in self.model.named_modules():
            if hasattr(mod, "loras_") and isinstance(mod.loras_, dict) and mod.loras_:
                yield name, mod

    def collect(self) -> Dict[str, Dict[int, nn.Module]]:
        """
        返回：
        {
          module_key: {
            0: lora_module_for_expert_0,
            1: lora_module_for_expert_1,
            ...
          },
          ...
        }
        """
        result: Dict[str, Dict[int, nn.Module]] = {}
        for mpath, mod in self.iter_lora_wrappers():
            bucket: Dict[int, nn.Module] = {}
            for lname, lmod in mod.loras_.items():
                if self.adapter_name_filter and not lname.startswith(self.adapter_name_filter):
                    # 限定只收某个 adapter（如 "moe.mixlora" / "mixlora_0"）
                    continue
                # 解析专家索引
                m = _EXPERT_NAME_RE.match(lname)
                if not m:
                    # 非专家型 LoRA（普通 LoRA），跳过
                    continue
                idx = int(m.group("idx"))
                bucket[idx] = lmod
            if bucket:
                result[_module_key_from_path(mpath)] = bucket
        return result


# ====== Merger ======

class ExpertMerger:
    """
    通用专家合并器：
    - 从模型收集多个专家的 A/B
    - 对齐 rank / 形状
    - 依据策略计算权重
    - 逐层/逐块加权合并，返回新的 LoRA 权重包
    """
    def __init__(self, spec: MergeSpec):
        self.spec = spec
        self.compute_dt = _tensor_dtype(spec)
        self.compute_dev = spec.device or torch.device("cpu")

    # ---- public API ----
    @torch.no_grad()
    def merge(self, model: nn.Module, adapter_prefix: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[int, float]]]:
        """
        adapter_prefix: 例如 "moe.mixlora" 或 "moe.mola"（与你实际注册专家的前缀一致）
        return:
          merged_weight_dict: { "module_key::(adapter_prefix+suffix)::A/B": tensor, ... }
          merge_report: { module_key: {expert_idx: weight, ...}, ... }
        """
        collector = LoRAExpertCollector(model, adapter_name_filter=adapter_prefix)
        experts_map = collector.collect()

        merged_weight_dict: Dict[str, torch.Tensor] = {}
        merge_report: Dict[str, Dict[int, float]] = {}

        target_adapter_name = adapter_prefix + self.spec.name_suffix

        for module_key, experts in experts_map.items():
            # 1) 筛选要参与合并的专家
            expert_ids = sorted(experts.keys())
            if self.spec.experts != "all":
                expert_ids = [i for i in expert_ids if i in set(self.spec.experts)]
            if len(expert_ids) == 0:
                continue

            # 2) 取出 A/B，并对齐到公共 rank（必要时）
            A_list, B_list, ranks = [], [], []
            for i in expert_ids:
                A, B = _extract_AB_from_lora_module(experts[i])
                A_list.append(_to_dtype(A.detach().to(self.compute_dev), self.compute_dt))
                B_list.append(_to_dtype(B.detach().to(self.compute_dev), self.compute_dt))
                ranks.append(A.shape[0])  # LoRA rank = rows of A

            A_list, B_list = self._align_rank(A_list, B_list, self.spec.target_rank)

            # 3) 计算合并权重
            weights = self._compute_weights(module_key, expert_ids)
            merge_report[module_key] = {eid: float(weights[eid]) for eid in expert_ids}

            # 4) 逐专家加权合并
            A_merged = torch.zeros_like(A_list[0])
            B_merged = torch.zeros_like(B_list[0])
            for eid, A_i, B_i in zip(expert_ids, A_list, B_list):
                w = weights[eid]
                A_merged.add_(A_i, alpha=w)
                B_merged.add_(B_i, alpha=w)

            # 5) 写入 Weight Dict（推理期软加载或导出使用）
            _put_AB_into_lora_weightdict(merged_weight_dict, module_key, target_adapter_name, A_merged, B_merged)

        return merged_weight_dict, merge_report

    # ---- internals ----

    def _align_rank(self, A_list: List[torch.Tensor], B_list: List[torch.Tensor],
                    target_rank: Optional[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        将不同 rank 的专家对齐到同一 rank：
        - 默认用 min_rank（或用户指定 target_rank）
        - 简化版对齐：截断到公共 rank（更鲁棒）；如需更强对齐可在此做 SVD/procrustes
        """
        ranks = [A.shape[0] for A in A_list]
        if target_rank is None:
            tgt = min(ranks)
        else:
            tgt = min(target_rank, min(ranks))

        if all(r == tgt for r in ranks):
            return A_list, B_list

        A_out, B_out = [], []
        for A, B in zip(A_list, B_list):
            if A.shape[0] == tgt:
                A_out.append(A)
                B_out.append(B)
            else:
                A_out.append(A[:tgt, :].contiguous())
                B_out.append(B[:, :tgt].contiguous())
        return A_out, B_out

    def _compute_weights(self, module_key: str, expert_ids: List[int]) -> Dict[int, float]:
        # 默认：均匀
        weights = {i: 1.0 for i in expert_ids}

        if self.spec.strategy == "mean":
            return _normalize(weights)

        if self.spec.strategy == "task_weighted":
            if not self.spec.task_weights:
                return _normalize(weights)
            # 仅保留 expert_ids 中的权重
            sw = {i: float(self.spec.task_weights.get(i, 0.0)) for i in expert_ids}
            return _normalize(sw)

        if self.spec.strategy == "fisher_weighted":
            # 需要外部提供 per-module per-expert 的 Fisher 权重
            if not self.spec.fisher_stats:
                return _normalize(weights)
            sw = {}
            for i in expert_ids:
                key = f"{module_key}::expert_{i}"
                sw[i] = float(self.spec.fisher_stats.get(key, 0.0))
            return _normalize(sw)

        if self.spec.strategy == "joint_svd":
            # 简化版：这里不做旋转对齐，只使用 _align_rank 截断到公共 rank
            # 如需更强：在此插入分块 SVD/子空间对齐再回写
            return _normalize(weights)

        # fallback
        return _normalize(weights)


# ====== High-level helpers ======

@torch.no_grad()
def merge_and_prepare_adapter(
    model: nn.Module,
    adapter_prefix: str,
    spec: MergeSpec,
) -> Tuple[str, Dict[str, torch.Tensor], Dict[str, Dict[int, float]]]:
    """
    统一入口：合并并返回 (merged_adapter_name, merged_weight_dict, merge_report)
    - merged_adapter_name: adapter_prefix + spec.name_suffix
    - merged_weight_dict: 可直接交给 model.init_adapter(lora_config, weight_dict) 的第二个参数使用，
      或者由上层封装写盘导出。
    """
    merger = ExpertMerger(spec)
    merged_weight, merge_report = merger.merge(model, adapter_prefix)
    merged_name = adapter_prefix + spec.name_suffix
    return merged_name, merged_weight, merge_report
