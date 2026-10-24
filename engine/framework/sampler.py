# Phase 8 — Sampler (framework_contracts.md)。
# TP rank0-only sampling + broadcast; temperature=0.0 → torch.argmax 贪婪。
import torch


class Sampler:
    """贪婪/温度采样。TP 下只有 rank0 采样再 broadcast (此 PP 变体 TP=1 直接采样)。"""

    def __init__(self, tp_group=None):
        self.tp_group = tp_group

    def sample(self, logits: torch.Tensor, sampling_params: dict) -> list[int]:
        """logits:[B, vocab] (或 [B, T, vocab] 取最后)。返回每序列一个 token id。"""
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        temperature = sampling_params.get("temperature", 0.0)
        top_p = sampling_params.get("top_p", 1.0)
        B = logits.shape[0]
        tokens = []
        for b in range(B):
            if temperature <= 0.0:
                # 贪婪: argmax
                tokens.append(int(torch.argmax(logits[b]).item()))
            else:
                probs = torch.softmax(logits[b] / temperature, dim=-1)
                if top_p < 1.0:
                    # top-p 截断
                    sorted_p, sorted_i = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_p, dim=0)
                    cutoff = (cum > top_p).nonzero()[0, 0].item() + 1
                    probs[sorted_i[cutoff:]] = 0.0
                    probs = probs / probs.sum()
                tokens.append(int(torch.multinomial(probs, 1).item()))
        return tokens
