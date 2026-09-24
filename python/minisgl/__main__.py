import os
import sys

# Process group: the skill's supervisor launches this engine as a background
# child and stops the whole tree with `kill -TERM -$PID`. A python started
# from a non-interactive script is not a process-group leader, so setsid()
# succeeds and PGID == PID; every mp.Process worker child (schedulers,
# tokenizer/detokenizer) inherits the group and dies with the parent. In the
# foreground (interactive terminal) python already leads a group, setsid()
# raises, and the fallback is to leave the group as-is.
try:
    os.setsid()
except (PermissionError, OSError):
    pass


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    return [a for a in argv if a != flag]


if "--selftest" in sys.argv[1:]:
    from .selftest import run_selftest

    sys.exit(run_selftest(_strip_flag(sys.argv[1:], "--selftest")))
elif "--bench" in sys.argv[1:]:
    from .bench import run_bench

    sys.exit(run_bench(_strip_flag(sys.argv[1:], "--bench")))
else:
    from .server import launch_server

    launch_server()
