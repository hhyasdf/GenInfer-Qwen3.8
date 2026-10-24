# Phase 12 — PP 进程组 + 边界 P2P send/recv (pp_communication_contracts.md)。
#
# 核心语义 (唯一契约来源):
#   - PP 边界 = 相邻 rank 的单向 P2P send/recv 激活流, **不是集合通信**。
#   - stage0 持 token_embd, 末 stage 持 output(LMHead); 中间 stage 只传
#     IntermediateTensors({"hidden_states","residual"})。模型有 MTP 块时边界增传
#     token_embd 键 (输入 token embedding, 首 rank 打包 / 末 rank MTP 块读取 —
#     llama.cpp graph_mtp 双图输入 inp->embd 语义, evo-001 R5 接线)。
#   - 边界禁 all_reduce / all_gather / broadcast (那是 TP 语义)。
#   - dst=(rank+1)%pp_size, src=(rank-1)%pp_size。
#   - world_size==1 (dist 未初始化) → send 直通返回 dict, recv 返回 None。
#
# Round-2 D-2 修复: 元数据协议对齐 vLLM parallel_state.py:640-690 send_object/recv_object —
#   pickle.dumps → uint8 载荷, int64 长度前缀, 全部走 CPU gloo 组; 张量走 device (nccl) 组
#   torch.distributed.send/recv。Round-1 用 dist.send_object_list/recv_object_list(meta=[])
#   在 gloo 上报 "16 vs 0" (发送 N 项 pickle 载荷 vs 接收端空列表协商不一致)。
#   空张量对称跳过: 发送端 numel()==0 不发送载荷 (parallel_state.py:846-849),
#   接收端必须对称地不接收载荷 (parallel_state.py:900-926) — 只建 empty 张量。
#   Round-1 只跳发送端不跳接收端 → 空张量必死锁 (潜伏缺陷)。
#   Round-3 设备放置修复 (L1 FAIL-2): 元数据元组补 device 字段 (只抓 device.type,
#   不抓 index — 对照 vllm TensorMetadata parallel_state.py:65 + 发送侧
#   device=value.device.type :77-86), recv 侧 torch.empty(..., device=元数据 device)
#   分配缓冲区 (对照 vllm :919-923) — Round-2 recv 侧缺 device → CPU 张量,
#   NCCL 组拒收 (RuntimeError: No backend type associated with device type cpu);
#   空张量 latent 路径 (:141) 一并补 device。
from __future__ import annotations

import pickle

import torch
import torch.distributed as dist


class PPError(RuntimeError):
    """所有 PP 违规的统一异常。消息含陷阱编号 (PP-COMM-00X)。"""


class IntermediateTensors:
    """stage 边界激活容器。必含键: "hidden_states", "residual" (可含额外键, 如 token_embd)。"""

    def __init__(self, tensors: dict):
        missing = {"hidden_states", "residual"} - set(tensors.keys())
        if missing:
            raise PPError(
                f"PP-COMM-005: IntermediateTensors 缺键 {sorted(missing)} — "
                f"边界必含 hidden_states+residual")
        self.tensors = tensors

    def to(self, device) -> "IntermediateTensors":
        return IntermediateTensors({k: v.to(device) for k, v in self.tensors.items()})


def make_empty_intermediate_tensors(hidden_size: int, device, dtype) -> IntermediateTensors:
    """(hidden_size:int, device, dtype) -> IntermediateTensors。
    hidden_states/residual/token_embd 各 shape (1, hidden_size)。
    R5 (Issue 8): 同步建 token_embd 占位 — 末 rank MTP 块跨边界读取的输入 token
    embedding (llama.cpp graph_mtp inp->embd); 模型无 MTP 块的调用方不发送此键。"""
    return IntermediateTensors({
        "hidden_states": torch.zeros(1, hidden_size, device=device, dtype=dtype),
        "residual": torch.zeros(1, hidden_size, device=device, dtype=dtype),
        "token_embd": torch.zeros(1, hidden_size, device=device, dtype=dtype),
    })


def _send_object(obj, dst: int, group) -> None:
    """pickle 对象发送: int64 长度前缀 + uint8 载荷, 走 CPU 组。
    对照 vLLM parallel_state.py:640-663 send_object。"""
    data = pickle.dumps(obj, protocol=4)
    size = torch.tensor([len(data)], dtype=torch.int64)
    dist.send(size, dst=dst, group=group)
    payload = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    dist.send(payload, dst=dst, group=group)


def _recv_object(src: int, group):
    """接收 _send_object 发送的 pickle 对象。对照 vLLM parallel_state.py:666-690 recv_object。"""
    size = torch.empty(1, dtype=torch.int64)
    dist.recv(size, src=src, group=group)
    payload = torch.empty(size.item(), dtype=torch.uint8)
    dist.recv(payload, src=src, group=group)
    return pickle.loads(payload.numpy().tobytes())


class PPGroup:
    """PP 进程组句柄。rank/pp_size/is_first/is_last 由 dist 环境注入 (init 时)。"""

    def __init__(self, rank: int, pp_size: int,
                 device_group=None, cpu_group=None):
        self.rank = rank
        self.pp_size = pp_size
        self.is_first_rank = (rank == 0)
        self.is_last_rank = (rank == pp_size - 1)
        self.device_group = device_group  # 默认 world (nccl)
        self.cpu_group = cpu_group        # gloo, 元数据 object 交换
        self._pending_sends = []

    # -- P2P 边界 --------------------------------------------------------
    def _norm_dst(self, dst):
        return (self.rank + 1) % self.pp_size if dst is None else dst

    def _norm_src(self, src):
        return (self.rank - 1) % self.pp_size if src is None else src

    def _global_peer(self, peer):
        return self._ranks[peer] if hasattr(self, "_ranks") else peer

    def send_tensor_dict(self, tensor_dict, dst: int | None = None,
                         all_gather_group=None, async_send: bool = False):
        """P2P send。非末 stage 在 forward 后调用。
        pp_size==1 (dist 未初始化) → 直通返回 dict。否则 P2P 发送, 返回 None。
        接受 dict 或 IntermediateTensors。
        空张量 (numel()==0): 元数据照发, 载荷跳过 (对称规则, 接收端不 recv)。"""
        if isinstance(tensor_dict, IntermediateTensors):
            tensor_dict = tensor_dict.tensors
        # 直通: 单 stage (无真实 PP 边界)
        if not dist.is_initialized() or self.pp_size == 1:
            return tensor_dict

        global_dst = self._global_peer(self._norm_dst(dst))
        # 元数据 (key, shape, dtype, device, is_tensor) 走 CPU gloo pickle 协议。
        # device 只抓 type (如 "cuda"/"cpu"), 不抓 index — 接收侧按本 rank 当前设备
        # 分配缓冲区 (vllm "receiving side will set the device index", :77-86 注释)。
        # R5: shape/dtype 提取与 device 一样按 is_tensor 防护 (R4 潜伏缺陷: 非 tensor
        # 值会先在 tuple(v.shape) 上 AttributeError, 防护只在 device 一行)。接收侧
        # 对 is_tensor=False 条目 continue, 与本防护对称。
        meta = [(k,
                 tuple(v.shape) if isinstance(v, torch.Tensor) else None,
                 v.dtype if isinstance(v, torch.Tensor) else None,
                 v.device.type if isinstance(v, torch.Tensor) else None,
                 isinstance(v, torch.Tensor))
                for k, v in tensor_dict.items()]
        _send_object(meta, global_dst, self.cpu_group)
        for k, v in tensor_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.numel() == 0:
                continue  # 对称跳过: 接收端据 numel==0 不 recv 载荷
            t = v.contiguous()
            if async_send:
                self._pending_sends.append(
                    dist.isend(t, dst=global_dst, group=self.device_group))
            else:
                dist.send(t, dst=global_dst, group=self.device_group)
        return None

    def recv_tensor_dict(self, src: int | None = None,
                         all_gather_group=None):
        """P2P recv。非首 stage 在 forward 前调用。
        pp_size==1 → 返回 None。否则 P2P 接收, 返回 IntermediateTensors。
        空张量 (numel()==0): 对称跳过载荷 recv, 只建 empty 张量。"""
        if not dist.is_initialized() or self.pp_size == 1:
            return None
        global_src = self._global_peer(self._norm_src(src))
        meta = _recv_object(global_src, self.cpu_group)
        tensors = {}
        for (k, shape, dtype, device, is_tensor) in meta:
            if not is_tensor:
                continue
            # R3: 缓冲区按元数据 device 分配 (vllm :919-923 范式)。"cuda" 不带 index
            # → 落在本 rank 当前设备 (NCCL 通信器绑定设备); "cpu" → gloo 组合法。
            n = int(torch.tensor(shape).prod().item()) if shape else 0
            if n == 0:
                tensors[k] = torch.empty(shape, dtype=dtype, device=device)
                continue  # 对称跳过: 发送端未发空张量载荷
            t = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(t, src=global_src, group=self.device_group)
            tensors[k] = t
        return IntermediateTensors(tensors)

    # -- 集合通信禁令 ----------------------------------------------------
    def all_reduce(self, tensor) -> None:
        """⛔ stage 内 / 边界调 all_reduce 必抛 PPError(PP-COMM-002)。"""
        raise PPError(
            "PP-COMM-002: all_reduce forbidden inside stage; "
            "PP boundary is P2P send/recv only, no collectives")

    def barrier(self) -> None:
        """全 stage 同步 (仅 init/teardown, 不进热路径)。用 CPU 组避免 NCCL barrier 副作用。"""
        if not dist.is_initialized() or self.pp_size == 1:
            return
        dist.barrier(group=self.cpu_group)

    def wait_sends(self):
        for req in self._pending_sends:
            req.wait()
        self._pending_sends = []


_PP_GROUP: PPGroup | None = None


def init_pp_process_group() -> PPGroup:
    """() -> PPGroup。幂等; rank/pp_size 由 dist 环境注入。
    dist 未初始化 → pp_size=1, rank=0 直通组 (单卡回退)。"""
    global _PP_GROUP
    if _PP_GROUP is not None:
        return _PP_GROUP
    if dist.is_initialized():
        rank = dist.get_rank()
        pp_size = dist.get_world_size()
        cpu_group = dist.new_group(backend="gloo")
        _PP_GROUP = PPGroup(rank=rank, pp_size=pp_size,
                            device_group=None, cpu_group=cpu_group)
        _PP_GROUP._ranks = list(range(pp_size))
    else:
        _PP_GROUP = PPGroup(rank=0, pp_size=1, device_group=None, cpu_group=None)
    return _PP_GROUP


def get_pp_group() -> PPGroup:
    """() -> PPGroup。未 init 则抛 PPError(PP-COMM-006), 不静默降级。"""
    if _PP_GROUP is None:
        raise PPError("PP-COMM-006: get_pp_group() called before init_pp_process_group()")
    return _PP_GROUP
