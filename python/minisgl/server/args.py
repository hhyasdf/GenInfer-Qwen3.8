from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


def _load_config(config_path: str) -> Dict[str, Any]:
    """Read ``config.toml`` (if present) and return its parsed contents.

    The gen-inference skill bakes the scenario's parameters into
    ``config.toml``. The engine reads it as its default source so the cmd
    entry point (``python -m minisgl``, zero arguments) works without the
    supervisor. CLI arguments and environment variables override the config
    values. Returns an empty dict when the file is absent (the engine then
    falls back to its built-in defaults).
    """
    if not os.path.isfile(config_path):
        return {}
    import tomllib

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _config_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map ``config.toml`` fields onto the argparse destination names."""
    defaults: Dict[str, Any] = {}
    engine = cfg.get("engine", {})
    if "model" in engine:
        defaults["model_path"] = engine["model"]
    if "dtype" in engine:
        defaults["dtype"] = engine["dtype"]
    if "tp" in engine:
        defaults["tensor_parallel_size"] = engine["tp"]
    model = cfg.get("model", {})
    if "max_context" in model:
        defaults["max_seq_len_override"] = model["max_context"]
    perf = cfg.get("performance", {})
    if "max_running_requests" in perf:
        defaults["max_running_req"] = perf["max_running_requests"]
    if "memory_ratio" in perf:
        defaults["memory_ratio"] = perf["memory_ratio"]
    if "page_size" in perf:
        defaults["page_size"] = perf["page_size"]
    if "attention_backend" in perf:
        defaults["attention_backend"] = perf["attention_backend"]
    if "cache_type" in perf:
        defaults["cache_type"] = perf["cache_type"]
    if "cuda_graph_max_bs" in perf:
        defaults["cuda_graph_max_bs"] = perf["cuda_graph_max_bs"]
    if "max_prefill_length" in perf:
        defaults["max_extend_tokens"] = perf["max_prefill_length"]
    serving = cfg.get("serving", {})
    if "host" in serving:
        defaults["server_host"] = serving["host"]
    if "port" in serving:
        defaults["server_port"] = serving["port"]
    distributed = cfg.get("distributed", {})
    if "layer_split" in distributed:
        defaults["layer_split"] = tuple(distributed["layer_split"])
    if "draft_device" in distributed:
        defaults["draft_device"] = distributed["draft_device"]
    return defaults


def _apply_sampling_env(cfg: Dict[str, Any]) -> None:
    """Set the ``MINISGL_SAMPLING_*`` environment variables from config.

    The sampling defaults are read from the environment by the API server.
    The supervisor sets these from ``config.toml``; the cmd entry point (zero
    arguments) does not go through the supervisor, so the engine sets them
    here. Existing environment variables are never overridden.
    """
    sampling = cfg.get("sampling", {})
    for key, env_var in (
        ("temperature", "MINISGL_SAMPLING_TEMPERATURE"),
        ("top_p", "MINISGL_SAMPLING_TOP_P"),
        ("top_k", "MINISGL_SAMPLING_TOP_K"),
    ):
        if key in sampling and env_var not in os.environ:
            os.environ[env_var] = str(sampling[key])
    model = cfg.get("model", {})
    if "max_new_tokens" in model and "MINISGL_SAMPLING_MAX_TOKENS" not in os.environ:
        os.environ["MINISGL_SAMPLING_MAX_TOKENS"] = str(model["max_new_tokens"])


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    # Read the baked-in config (config.toml) as the default source. The
    # gen-inference skill bakes the scenario's parameters into config.toml;
    # the engine reads it so the cmd entry point (``python -m minisgl``, zero
    # arguments) works without the supervisor. CLI arguments override the
    # config values.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.toml")
    pre_args, _ = pre_parser.parse_known_args(args)
    cfg = _load_config(pre_args.config)
    _apply_sampling_env(cfg)
    cfg_defaults = _config_defaults(cfg)

    def d(key: str, fallback: Any) -> Any:
        """Config value for ``key`` if present, else ``fallback``."""
        return cfg_defaults.get(key, fallback)

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default=d("model_path", None),
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID. "
        "Defaults to the [engine] model field in config.toml.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=d("dtype", "auto"),
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=d("tensor_parallel_size", 1),
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--layer-split",
        type=str,
        default=d("layer_split", None),
        help="Non-uniform layer split across ranks, e.g. '26,38' (activation "
        "handoff, not TP). Rank i runs the i-th block of layers; the front rank "
        "owns the embedding and the back rank owns the lm_head.",
    )

    parser.add_argument(
        "--draft-device",
        type=int,
        default=d("draft_device", None),
        help="The cuda device index that hosts the draft model (speculative decoding).",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=d("max_running_req", ServerArgs.max_running_req),
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=d("max_seq_len_override", ServerArgs.max_seq_len_override),
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=d("memory_ratio", ServerArgs.memory_ratio),
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=d("server_host", ServerArgs.server_host),
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=d("server_port", ServerArgs.server_port),
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=d("cuda_graph_max_bs", ServerArgs.cuda_graph_max_bs),
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=d("max_extend_tokens", ServerArgs.max_extend_tokens),
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=d("page_size", ServerArgs.page_size),
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=d("attention_backend", ServerArgs.attention_backend),
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=d("cache_type", ServerArgs.cache_type),
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs.pop("config")  # the config path is not part of ServerArgs
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"] is None:
        parser.error(
            "--model-path is required (or the [engine] model field in config.toml)"
        )
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str

    # Parse the layer split (a tuple of ints from config.toml, or a comma
    # string from the CLI). The layer split is not TP: the world size is the
    # number of ranks, and each rank binds to cuda:{rank}.
    if kwargs.get("layer_split") is not None:
        ls = kwargs["layer_split"]
        if isinstance(ls, str):
            ls = [int(x) for x in ls.split(",") if x.strip()]
        if len(ls) < 2:
            parser.error("--layer-split needs at least two ranks, e.g. '26,38'")
        kwargs["layer_split"] = tuple(ls)
        kwargs["tensor_parallel_size"] = len(ls)
    else:
        kwargs.pop("layer_split", None)

    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
