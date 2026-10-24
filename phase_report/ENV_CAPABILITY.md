# Environment Capability Report

- Timestamp: 2026-10-18T13:25:00+08:00
- GPU: GPU0 = NVIDIA GeForce RTX 4070 Ti SUPER (sm89, 16 GB); GPU1 = NVIDIA GeForce RTX 2080 Ti (sm75, 20 GB)。同构? 否（异构双卡）。NVLink? 无（WSL，nvidia-smi topo 不可用 → PCIe）。
- vllm._custom_ops: AVAILABLE
- flash_attn: UNAVAILABLE (ModuleNotFoundError: No module named 'flash_attn') → attention 用 F.scaled_dot_product_attention（SAFE）或手动 matmul+softmax 回退
- F.scaled_dot_product_attention: SAFE (1,1,64,64 fp16 实测)
- Parallel strategy: PP=2 (layer-split，层切分流水线)
- Parallel strategy 依据: Step 6.5 规则 R3 命中。W≈23.8GB（Q6_K 主 gguf 23.18GB + mmproj Q8_0 0.63GB）；L=20GB（2080 Ti），S=16GB（4070 Ti SUPER）；R1: 23.8 > 0.75×20=15 → 否；R2: W/N=11.9GB > 0.5×16=8GB 且异构 → 否；R3 → PP。异构非均分切层（GPU1 多承担约 1/4 层数），层边界须对齐 full_attention_interval=4 与 MTP 层。
- 模型架构（GGUF 头直接提取）: arch=qwen35, block_count=65(64+MTP1), embedding=5120, ffn=17408, heads=24, kv_heads=4 (GQA 6:1), head_dim=256, full_attention_interval=4 (16 full-attn + 48 GDN), nextn_predict_layers=1, ssm(conv=4, state=128, groups=16, ts_rank=48, inner=6144), rope(dim=64, 3 valid sections [11,11,10] + 0 padding, freq_base=10000000.0), rms_eps=1e-6, vocab=248320, tokenizer/chat_template 内嵌 GGUF。文件特性：标量值 4 字节对齐填充（int16+2B 零）；array_type 用值类型码（8=字符串数组）。llama.cpp gguf 读取器原生支持（llama-server 正在加载此文件）。
