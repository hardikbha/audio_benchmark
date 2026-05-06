import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple


def _parse_visible_gpu_env() -> Optional[List[int]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            values.append(int(token))
    return values or None


def _query_gpu_free_memory() -> List[Tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpus: List[Tuple[int, int]] = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        idx_text, free_mem_text = [x.strip() for x in line.split(",", maxsplit=1)]
        gpus.append((int(idx_text), int(free_mem_text)))
    return gpus


def get_best_gpu(min_free_mib: int = 0, require_min_free: bool = False) -> Optional[int]:
    try:
        gpus = _query_gpu_free_memory()
        if not gpus:
            return None

        visible = _parse_visible_gpu_env()
        if visible is not None:
            gpus = [item for item in gpus if item[0] in visible]
            if not gpus:
                return None

        # Prioritize max free memory.
        gpus.sort(key=lambda x: x[1], reverse=True)

        if min_free_mib > 0:
            qualified = [item for item in gpus if item[1] >= min_free_mib]
            if qualified:
                return qualified[0][0]
            if require_min_free:
                return None

        return gpus[0][0]
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick GPU index with most free memory.")
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=0,
        help="Minimum free GPU memory in MiB.",
    )
    parser.add_argument(
        "--require-min-free",
        action="store_true",
        help="Exit non-zero if no GPU satisfies --min-free-mib.",
    )
    args = parser.parse_args()

    gpu_idx = get_best_gpu(
        min_free_mib=max(0, int(args.min_free_mib)),
        require_min_free=bool(args.require_min_free),
    )
    if gpu_idx is None:
        # Backward-compatible behavior: print 0 unless strict mode is requested.
        if args.require_min_free:
            return 2
        print("0")
        return 0

    print(gpu_idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
