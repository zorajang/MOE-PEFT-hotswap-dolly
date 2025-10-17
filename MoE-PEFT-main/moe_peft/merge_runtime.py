# moe_peft/merge_runtime.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

# 依赖第 1 步中新增的专家合并实现
from .adapters.merging import MergeSpec, merge_and_prepare_adapter

# 你的工程里已有 AdapterConfig / LLMModel 接口
try:
    from . import AdapterConfig  # moe_peft.AdapterConfig
except Exception:
    AdapterConfig = None


def build_merge_spec_from_dict(cfg: Dict[str, Any]) -> MergeSpec:
    """
    将模板/配置里的 expert_merging dict 转为 MergeSpec。
    未提供的键采用 MergeSpec 默认值。
    """
    if cfg is None:
        return MergeSpec()
    return MergeSpec(
        strategy      = cfg.get("strategy",      "mean"),
        scope         = cfg.get("scope",         "layerwise"),
        experts       = cfg.get("experts",       "all"),
        alpha         = cfg.get("alpha",         0.5),
        task_weights  = cfg.get("task_weights"),
        fisher_stats  = cfg.get("fisher_stats"),
        target_rank   = cfg.get("target_rank"),
        name_suffix   = cfg.get("name_suffix",   "-merged"),
        dtype         = getattr(torch, cfg.get("dtype"), None) if isinstance(cfg.get("dtype"), str) else cfg.get("dtype"),
        device        = torch.device(cfg["device"]) if isinstance(cfg.get("device"), str) else cfg.get("device"),
    )


@torch.no_grad()
def soft_merge_adapter(
    model: nn.Module,
    adapter_prefix: str,
    merge_spec: MergeSpec | Dict[str, Any],
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """
    执行“运行期软合并”，返回：
      - merged_adapter_name
      - merged_weight_dict（可直接用于 init_adapter 挂载）
    不写盘，不修改模型现有权重。
    """
    spec = build_merge_spec_from_dict(merge_spec) if isinstance(merge_spec, dict) else merge_spec
    merged_name, merged_weight, _ = merge_and_prepare_adapter(
        model=model,
        adapter_prefix=adapter_prefix,
        spec=spec,
    )
    return merged_name, merged_weight


@torch.no_grad()
def attach_merged_adapter(
    model: nn.Module,
    merged_adapter_name: str,
    merged_weight_dict: Dict[str, torch.Tensor],
):
    """
    将已合并的适配器**临时挂载**到模型上（不覆盖原有 adapter）。
    挂载后即可通过你的 generate 流程选择该 adapter_name 推理。
    """
    if AdapterConfig is None:
        raise RuntimeError("AdapterConfig not found in moe_peft; cannot attach merged adapter.")
    cfg = AdapterConfig(adapter_name=merged_adapter_name)
    # 你的 model.init_adapter 支持 (config, weight_dict) 二元调用
    model.init_adapter(cfg, merged_weight_dict)


@torch.no_grad()
def maybe_soft_merge_and_attach(
    model: nn.Module,
    adapter_prefix: Optional[str],
    expert_merging_cfg: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    便捷入口：
      - 当 expert_merging_cfg 启用时，现场合并并挂载，返回 merged_adapter_name；
      - 否则返回 None（不做任何事）。
    """
    if not expert_merging_cfg or not expert_merging_cfg.get("enabled", False):
        return None
    if not adapter_prefix:
        # 如未显式传入，可在外层按需实现自动推断；此处保持显式以避免误挂载
        raise ValueError("adapter_prefix is required for expert soft-merge.")
    merged_name, merged_w = soft_merge_adapter(
        model=model,
        adapter_prefix=adapter_prefix,
        merge_spec=expert_merging_cfg,
    )
    attach_merged_adapter(model, merged_name, merged_w)
    return merged_name
