from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn

from .merge_runtime import soft_merge_adapter
from .common import AdapterConfig


def hot_plug_merge(
    model: nn.Module,
    adapter_prefix: str,
    expert_merging_cfg: Dict[str, Any],
    *,
    override_adapter_name: Optional[str] = None,
) -> str:

    merged_name, merged_weight = soft_merge_adapter(
        model=model, adapter_prefix=adapter_prefix, merge_spec=expert_merging_cfg
    )

    target_name = override_adapter_name or merged_name
    cfg = AdapterConfig(adapter_name=target_name)
    # 使用 (config, weight_dict) 的二元调用进行挂载
    model.init_adapter(cfg, merged_weight)
    return target_name


def hot_unplug(model: nn.Module, adapter_name: str) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """
    从模型卸载指定适配器。返回 (lora_config, lora_weight) 方便后续复挂。
    """
    return model.unload_adapter(adapter_name)


class AdapterHotplugManager:
    """
    简单的热插拔管理器：管理当前已挂载/由本管理器创建的合并适配器。
    - 提供热插入/热拔出
    - 可查询已注册的合并适配器名称
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._owned_adapters: List[str] = []

    @torch.no_grad()
    def hot_plug_merge(
        self,
        adapter_prefix: str,
        expert_merging_cfg: Dict[str, Any],
        *,
        override_adapter_name: Optional[str] = None,
    ) -> str:
        name = hot_plug_merge(
            model=self.model,
            adapter_prefix=adapter_prefix,
            expert_merging_cfg=expert_merging_cfg,
            override_adapter_name=override_adapter_name,
        )
        if name not in self._owned_adapters:
            self._owned_adapters.append(name)
        return name

    @torch.no_grad()
    def hot_unplug(self, adapter_name: str) -> Tuple[Any, Dict[str, torch.Tensor]]:
        cfg, weights = hot_unplug(self.model, adapter_name)
        if adapter_name in self._owned_adapters:
            self._owned_adapters.remove(adapter_name)
        return cfg, weights

    def list_owned(self) -> List[str]:
        return list(self._owned_adapters)


