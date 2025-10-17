import fire
import torch

import moe_peft
from typing import Optional, Dict, Any

try:
    from moe_peft.merge_runtime import maybe_soft_merge_and_attach
except Exception as _e:
    maybe_soft_merge_and_attach = None  # 允许在未引入第三步代码时静默跳过


def inference_callback(cur_pos, outputs):
    print(f"Position: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} output: {output[0]}")


def _infer_expert_adapter_prefix_from_model(model) -> Optional[str]:
    """
    扫描模型内部模块，推断专家型 LoRA 的 adapter 前缀（如 'moe.mixlora' / 'moe.mola' / 'moe.loramoe'）。
    命名假设：存在 key 形如 'moe.<name>.experts.<idx>'。
    """
    for _, mod in model.named_modules():
        if hasattr(mod, "loras_") and isinstance(mod.loras_, dict) and mod.loras_:
            for lname in mod.loras_.keys():
                if ".experts." in lname and lname.startswith("moe."):
                    return lname.rsplit(".experts.", 1)[0]
    return None


def maybe_apply_expert_soft_merge(
    model,
    expert_merging_cfg: Optional[Dict[str, Any]] = None,
    adapter_prefix: Optional[str] = None,
) -> Optional[str]:
    """
    若传入 expert_merging_cfg 且启用，则现场合并并挂载，返回合并后的 adapter 名称；否则返回 None。
    """
    if maybe_soft_merge_and_attach is None:
        return None
    if not expert_merging_cfg or not expert_merging_cfg.get("enabled", False):
        return None

    prefix = adapter_prefix or _infer_expert_adapter_prefix_from_model(model)
    if prefix is None:
        # 无专家型 LoRA，忽略
        return None

    merged_name = maybe_soft_merge_and_attach(
        model=model,
        adapter_prefix=prefix,
        expert_merging_cfg=expert_merging_cfg,
    )
    return merged_name
# ==== /generate.py hook ====


def main(
    base_model: str,
    instruction: str,
    input: str = None,
    template: str = None,
    lora_weights: str = None,
    load_16bit: bool = True,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    max_seq_len: int = None,
    stream: bool = False,
    device: str = moe_peft.executor.default_device_name(),
):

    model = moe_peft.LLMModel.from_pretrained(
        base_model,
        device=device,
        attn_impl="flash_attn" if flash_attn else "eager",
        bits=(8 if load_8bit else (4 if load_4bit else None)),
        load_dtype=torch.bfloat16 if load_16bit else torch.float32,
    )
    tokenizer = moe_peft.Tokenizer(base_model)

    if lora_weights:
        adapter_name = model.load_adapter(lora_weights)
    else:
        adapter_name = model.init_adapter(
            moe_peft.AdapterConfig(adapter_name="default")
        )

    generate_paramas = moe_peft.GenerateConfig(
        adapter_name=adapter_name,
        prompt_template=template,
        prompts=[(instruction, input)],
    )

    output = moe_peft.generate(
        model,
        tokenizer,
        [generate_paramas],
        max_gen_len=max_seq_len,
        stream_callback=inference_callback if stream else None,
    )

    for prompt in output[adapter_name]:
        print(f"\n{'=' * 10}\n")
        print(prompt)
        print(f"\n{'=' * 10}\n")







if __name__ == "__main__":
    fire.Fire(main)
