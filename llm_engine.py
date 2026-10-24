# Phase 9 — LLMEngine 主循环 (engine_contracts.md)。
#
# 精确属性面 (gate 导入):
#   LLMEngine(model_dir: Path, inference_backend='qwen_tp', max_num_seqs=4)
#   attrs: block_size, device, _waiting, _running, _active_gen_seqs,
#          scheduler ( .schedule(waiting,running,num_free)→{is_prefill,batch},
#                      .postprocess(batch,True,tokens) ),
#          runner  ( .run(batch,is_prefill,temperature)→{next_tokens},
#                    .get_num_free_blocks() )
#   method: generate(prompt, max_new_tokens=N, temperature=0.0) → str
#
# PP 变体: 按 get_pp_indices 取本 rank 层区间建 stage-local 模型; 首 stage 持 token_embd,
# 末 stage 持 LMHead; 中间传 IntermediateTensors。权重 PP-aware 绑定 (只本区间层)。
import os
from pathlib import Path

import torch

from engine.models import QwenTPConfig, Qwen3ForCausalLMTP
from engine.pp_layer_partition import (
    compute_pp_layer_partition, compute_pp_layer_partition_budgeted,
    check_partition_budget, get_stage_layers, assign_mtp_device,
)
from engine.pp import init_pp_process_group, get_pp_group
from engine.framework import Scheduler, ModelRunner, Sequence, SequenceStatus
from engine.weights import PPWeightLoader
from engine.tokenizer import load_tokenizer


class LLMEngine:
    """PP 推理引擎主循环。model_dir 含 config.json + 主 GGUF + mmproj GGUF。"""

    def __init__(self, model_dir, inference_backend="qwen_tp", max_num_seqs=4):
        self.model_dir = Path(model_dir)
        self.inference_backend = inference_backend
        self.max_num_seqs = max_num_seqs
        # 框架常量
        self.block_size = 256
        self._waiting: list = []
        self._running: list = []
        self._active_gen_seqs: dict = {}
        self.scheduler = Scheduler(block_size=self.block_size)
        # PP 坐标 (环境变量 PP_RANK/PP_SIZE, 缺省单进程 pp_size=1)
        self.pp_rank = int(os.environ.get("PP_RANK", os.environ.get("RANK", "0")))
        self.pp_size = int(os.environ.get("PP_SIZE", os.environ.get("WORLD_SIZE", "1")))
        # config + 分区
        self.cfg = QwenTPConfig.from_pretrained(str(self.model_dir))
        num_layers = self.cfg.num_hidden_layers
        # R4 fix ②: vram_ratio 切层 + 整 4 维 VRAM 预算校验 (W 权重 + D dequant 临时峰
        # + A 激活峰 + K KV 配额 + G GDN 状态; 越安全线 → PP-PART-006, pp_size=2 自动重推
        # FA 对齐边界)。evo-001 的 28/36 满足两卡安全线 (stage0 ~10.01GiB ≤ 15.1 /
        # stage1 ~14.99GiB ≤ 21.1) → 原样保留。
        self.partition = compute_pp_layer_partition_budgeted(
            num_layers, self.pp_size, vram_ratio=self._vram_ratio())
        self.stage_vram = check_partition_budget(self.partition)  # 预算明细 (内省 / 日志)
        self.stage_layers = get_stage_layers(num_layers, self.pp_rank, self.pp_size,
                                             partition_list=self.partition)
        self._mtp_device = assign_mtp_device(self.partition, self.cfg.mtp_layers[0],
                                             self.pp_size) if self.cfg.mtp_layers else None
        # 设备: 本 rank 的 GPU (CUDA_VISIBLE_DEVICES 隔离), 无则 cpu
        self.device = self._resolve_device()
        # 懒建 (import 自检不占内存)
        self._model = None
        self._runner = None
        self._tok = None
        self._loaded = False

    def _resolve_device(self):
        if torch.cuda.is_available():
            return f"cuda:{self.pp_rank % torch.cuda.device_count()}"
        return "cpu"

    def _vram_ratio(self) -> float:
        """末段/首段 显存比 (vram_ratio)。两卡显存可探测时按 total_memory 计算, 否则 1.0。
        D-3: 异构双卡 (如 15.1/21.1 GiB) → ratio≈1.4 → 大卡分更多层。"""
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() >= self.pp_size and self.pp_size >= 2:
                first = torch.cuda.get_device_properties(0).total_memory
                last = torch.cuda.get_device_properties(self.pp_size - 1).total_memory
                if first > 0:
                    return last / first
        except Exception:
            pass
        return 1.0

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    @property
    def runner(self) -> ModelRunner:
        if self._runner is None:
            self._build_model()
            self._runner = ModelRunner(self._model, self.scheduler,
                                       block_size=self.block_size,
                                       max_blocks=160, device=self.device)
        return self._runner

    def _build_model(self):
        init_pp_process_group()
        model = Qwen3ForCausalLMTP(
            self.cfg, self.pp_rank, self.pp_size,
            list(self.stage_layers), device=self.device)
        self._model = model
        # PP-aware 权重绑定 (只本区间层)
        gguf = self._find_main_gguf()
        loader = PPWeightLoader(model, gguf, list(self.stage_layers),
                                is_first=(self.pp_rank == 0),
                                is_last=(self.pp_rank == self.pp_size - 1),
                                device=self.device)
        self._n_loaded = loader.load()
        self._loaded = True
        model.to(self.device).eval()

    def _find_main_gguf(self) -> Path:
        """主模型 GGUF (排除 mmproj)。"""
        for p in self.model_dir.glob("*.gguf"):
            if "mmproj" not in p.name.lower():
                return p
        raise FileNotFoundError(f"no main GGUF in {self.model_dir}")

    # --- 请求生命周期 ---
    def add_request(self, prompt, token_ids=None, max_new_tokens=64, temperature=0.0):
        if token_ids is None:
            token_ids = self._tokenize(prompt)
        seq = Sequence(self._next_id(), token_ids)
        seq.sampling_params = {"max_tokens": max_new_tokens,
                               "temperature": temperature, "top_p": 1.0}
        self._waiting.append(seq)
        return seq

    def _next_id(self):
        self._id_counter = getattr(self, "_id_counter", 0) + 1
        return f"req-{self._id_counter}"

    @property
    def _tokenizer(self):
        """惰性加载真实 BPE tokenizer (D-6b)。从主 GGUF 内嵌 tokenizer.ggml 构建。"""
        if self._tok is None:
            self._tok = load_tokenizer(str(self._find_main_gguf()))
        return self._tok

    def _tokenize(self, prompt: str) -> list[int]:
        """真实 BPE tokenize (qwen35 pre + GPT-2 byte fallback)。对照 llama-server /tokenize。"""
        return self._tokenizer.tokenize(prompt)

    def _detokenize(self, token_ids) -> str:
        """真实 BPE detokenize。detokenize(tokenize(x)) == x 闭包。"""
        return self._tokenizer.detokenize(list(token_ids))

    # --- 主循环 ---
    def step(self) -> bool:
        """一步调度+执行。返回是否有进展。"""
        num_free = self.runner.get_num_free_blocks()
        result = self.scheduler.schedule(self._waiting, self._running, num_free)
        batch, is_prefill = result["batch"], result["is_prefill"]
        if not batch:
            return False
        out = self.runner.run(batch, is_prefill,
                              temperature=batch[0].sampling_params.get("temperature", 0.0))
        self.scheduler.postprocess(batch, True, out["next_tokens"])
        # 移出完成的
        self._running = [s for s in self._running if not s.is_finished()]
        self._waiting = [s for s in self._waiting
                         if s.status == SequenceStatus.WAITING]
        return True

    def generate(self, prompt, max_new_tokens=64, temperature=0.0) -> str:
        """贪婪生成, 返回完整文本。"""
        seq = self.add_request(prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature)
        while not seq.is_finished():
            if not self.step():
                break
            if seq.is_finished():
                break
        return self._detokenize(seq.output_ids)
