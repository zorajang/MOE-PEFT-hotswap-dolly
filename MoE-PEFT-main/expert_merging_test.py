# # -*- coding: utf-8 -*-
# import os, json, shutil
# import torch

# import moe_peft as mp
# from moe_peft import LLMModel, adapter_factory
# from moe_peft.adapters import lora_config_factory  # 用于把dict转成配置对象

# # --------- 1) 基础路径（按你的真实路径 B） ----------
# BASE = r"D:\hf-cache\models--mistralai--Mistral-7B-v0.1\snapshots\27d67f1b5f57dc0953326b2601d68371d40ea8da"

# # 方便你切换不同模板：mixlora_dynamic.json / mixlora.json / loramoe.json / ...
# TEMPLATE_JSON = os.path.join(os.getcwd(), "templates", "mixlora_dynamic.json")  # 你项目里的 templates

# # --------- 2) 加载 Mistral-7B（本地离线） ----------
# def load_llm(local_root):
#     # 直接用模型根目录（不是单个config.json）
#     name_or_path = local_root
#     model = mp.LLMModel.from_pretrained(
#         name_or_path=name_or_path,
#         device="cpu",
#         bits=0,                 # 不量化，避免显存紧张可以换成 8/4
#         load_dtype=torch.float16,
#         compute_dtype=torch.float16,
#         attn_impl="eager",
#         use_sliding_window=False,
#     )
#     return model

# # --------- 3) 从模板构造 AdapterConfig 对象 ----------
# def make_adapter_configs(template_path, base_name="mixlora"):
#     with open(template_path, "r", encoding="utf-8") as f:
#         obj = json.load(f)

#     # 模板结构：{"lora":[{...一个配置...}]}，我们克隆两份做 exp_a / exp_b
#     base_cfg = obj["lora"][0].copy()
#     # 给第一、第二个专家起不同名字
#     cfg_a = base_cfg.copy();  cfg_a["name"] = f"{base_name}_exp_a"
#     cfg_b = base_cfg.copy();  cfg_b["name"] = f"{base_name}_exp_b"

#     # 用官方工厂把 dict 变成配置对象（会自动识别 routing_strategy / peft_type）
#     a = lora_config_factory(cfg_a)   # -> MixLoraConfig/LoraConfig/...
#     b = lora_config_factory(cfg_b)
#     return a, b

# # --------- 4) 软合并逻辑（简单平均），输出新 adapter 名 ----------
# def soft_merge(model: LLMModel, expert_names, merged_name="merged", strategy="mean"):
#     assert strategy in ["mean"], "当前示例只实现 simple mean"
#     # 取第一个专家的结构当模板
#     base_name = expert_names[0]
#     weight_dicts = []
#     for name in expert_names:
#         wd = model.get_adapter_weight_dict(name)   # 官方方法提取 adapter 的权重字典
#         weight_dicts.append(wd)

#     # 简单逐键平均
#     merged_w = {}
#     for k in weight_dicts[0].keys():
#         merged_w[k] = sum(wd[k] for wd in weight_dicts) / float(len(weight_dicts))

#     # 用第一个专家的配置作为新配置的骨架
#     base_cfg = model.adapter_configs_[base_name]
#     # 复制一个配置对象并改名
#     new_cfg = type(base_cfg).from_config(base_cfg.export() if hasattr(base_cfg, "export") else {
#         "name": base_cfg.adapter_name,
#         "task_name": getattr(base_cfg, "task_name", "casual"),
#         # LoRA参数落在 .from_config 中会被读取（见 LoraConfig.from_config）
#     })
#     new_cfg.adapter_name = merged_name

#     # 在模型中注册新 adapter 并加载合并后的权重
#     model.init_adapter(new_cfg, merged_w)          # 官方 API：传配置 + 权重【见源码签名】
#     return merged_name

# # --------- 5) 保存工具 ----------
# def save_adapter(model: LLMModel, adapter_name: str, out_dir: str):
#     os.makedirs(out_dir, exist_ok=True)
#     # 保存权重 + 配置（最简方式：把 state_dict 存起来，再把 config.export() 存到 json）
#     w = model.get_adapter_weight_dict(adapter_name)
#     torch.save(w, os.path.join(out_dir, f"{adapter_name}.pt"))

#     cfg = model.adapter_configs_[adapter_name]
#     cfg_json = cfg.export() if hasattr(cfg, "export") else {"name": cfg.adapter_name, "task_name": cfg.task_name}
#     with open(os.path.join(out_dir, f"{adapter_name}.json"), "w", encoding="utf-8") as f:
#         json.dump(cfg_json, f, ensure_ascii=False, indent=2)

# def main():
#     model = load_llm(BASE)

#     # 读取模板 -> 生成两个专家配置
#     cfg_a, cfg_b = make_adapter_configs(TEMPLATE_JSON, base_name="mixlora")

#     # 空初始化挂载两个专家（不提供 weight 字典）
#     model.init_adapter(cfg_a)   # 正确签名：需要 AdapterConfig 子类实例【:contentReference[oaicite:7]{index=7}】
#     model.init_adapter(cfg_b)

#     # 做一次“软合并”
#     merged = soft_merge(model, [cfg_a.adapter_name, cfg_b.adapter_name], merged_name="merged")

#     # 保存三个适配器以便检查
#     out_root = os.path.join(os.getcwd(), "expert_merging_out")
#     if os.path.exists(out_root):
#         shutil.rmtree(out_root)
#     os.makedirs(out_root, exist_ok=True)

#     save_adapter(model, cfg_a.adapter_name, os.path.join(out_root, cfg_a.adapter_name))
#     save_adapter(model, cfg_b.adapter_name, os.path.join(out_root, cfg_b.adapter_name))
#     save_adapter(model, merged,             os.path.join(out_root, merged))

#     print("\nDone. Adapters saved to:", out_root)

# if __name__ == "__main__":
#     main()




# 诊断 + 修复 + 一句生成
import os, json, torch, moe_peft
from moe_peft import Tokenizer, GenerateConfig
from moe_peft.adapters import lora_config_factory

BASE = r"D:\hf-cache\models--mistralai--Mistral-7B-v0.1\snapshots\27d67f1b5f57dc0953326b2601d68371d40ea8da"
MERGED_DIR = r".\expert_merging_out\merged"

# 1) 基础检查
assert os.path.isfile(os.path.join(MERGED_DIR, "merged.json")), "找不到 merged.json"
assert os.path.isfile(os.path.join(MERGED_DIR, "merged.pt")),   "找不到 merged.pt"

# 2) 载模型（CPU，避免 OOM）
model = moe_peft.LLMModel.from_pretrained(BASE, device="cpu", attn_impl="eager", use_sliding_window=False)
tok = Tokenizer(BASE)

# 3) 读取配置并确保 adapter_name 正确
cfg_dict = json.load(open(os.path.join(MERGED_DIR, "merged.json"), encoding="utf-8"))
print("merged.json keys ->", list(cfg_dict.keys()))
cfg = lora_config_factory(cfg_dict)

# 如果名字缺失/为 None，强制设置为 'merged'
if getattr(cfg, "adapter_name", None) in (None, "", "default"):
    cfg.adapter_name = cfg_dict.get("name") or "merged"
print("adapter_name ->", cfg.adapter_name)

# 4) 加载权重并挂载
w = torch.load(os.path.join(MERGED_DIR, "merged.pt"), map_location="cpu", weights_only=True)
model.init_adapter(cfg, w)

# 5) 确认适配器列表 & 简短生成
print("adapters loaded:", list(model.adapter_configs_.keys()))
outs = moe_peft.generate(
    model=model, tokenizer=tok,
    configs=[GenerateConfig(adapter_name=cfg.adapter_name, prompts=["Hello, merge check."])],
    max_gen_len=8,
)
print("OUTPUT:", outs[cfg.adapter_name][0])



