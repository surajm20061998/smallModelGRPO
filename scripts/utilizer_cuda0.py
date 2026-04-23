import time
from datetime import datetime
import subprocess
import sys


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
    gpu_index = 0
    while True:
        try:
            util, mem_used, mem_total = query_gpu(gpu_index)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] cuda:0 util : {util}% | mem : {mem_used} MiB / {mem_total} MiB",
                flush=True,
            )
        except Exception as exc:
            print(f"cuda:0 util : query_failed ({exc})", file=sys.stderr, flush=True)
        time.sleep(600)


if __name__ == "__main__":
    main()
