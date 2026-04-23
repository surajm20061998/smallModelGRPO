import argparse
import subprocess
import sys
import time
from datetime import datetime


def query_gpu(gpu_index: int) -> tuple[int, int, int]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True).strip()
    util_str, mem_used_str, mem_total_str = [part.strip() for part in output.split(",")]
    return int(util_str), int(mem_used_str), int(mem_total_str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()

    label = f"cuda:{args.gpu_index}"

    while True:
        try:
            util, mem_used, mem_total = query_gpu(args.gpu_index)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] {label} util : {util}% | mem : {mem_used} MiB / {mem_total} MiB",
                flush=True,
            )
        except Exception as exc:
            print(f"{label} util : query_failed ({exc})", file=sys.stderr, flush=True)

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
