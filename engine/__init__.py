# engine — Qwen3.5 hybrid 推理框架 (PP track)。
# 子包: kernels (数值基元), pp / pp_layer_partition / pp_kv / pp_scheduler (PP 契约),
#       tp_layers + tp_comm (TP no-op guard 接口), models (config + 模型),
#       weights (GGUF 反量化 + PP-aware 绑定), framework (sequence/scheduler/sampler/runner)。
