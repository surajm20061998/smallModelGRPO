#!/bin/bash

# GPU Keepalive Script
# Runs GPU workload every 5 minutes to prevent instance termination

GPU_KEEPALIVE_PID=""
INTERVAL_SECONDS=300
WORKLOAD_DURATION=30

trap 'echo "Shutting down GPU keepalive..."; kill $GPU_KEEPALIVE_PID 2>/dev/null; wait $GPU_KEEPALIVE_PID 2>/dev/null; exit 0' EXIT INT TERM

echo "Starting periodic GPU keepalive on cuda:0..."
echo "Workload will run for ${WORKLOAD_DURATION}s every ${INTERVAL_SECONDS}s (5 minutes)"

run_keepalive_workload() {
    local duration=$1
    KEEPALIVE_LOG=$(mktemp)
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU workload for ${duration}s..."
    
    uv run python -c "
import signal, sys, time, torch
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
t = torch.randn(4096, 4096, device='cuda:0', dtype=torch.bfloat16)
print('GPU keepalive running on cuda:0...', flush=True)
end_time = time.time() + $duration
while time.time() < end_time:
    torch.matmul(t, t)
    time.sleep(0.0005)
print('GPU keepalive workload completed.', flush=True)
" > "$KEEPALIVE_LOG" 2>&1 &
    
    GPU_KEEPALIVE_PID=$!
    
    until grep -q "GPU keepalive running" "$KEEPALIVE_LOG" 2>/dev/null; do
        sleep 0.5
    done
    
    wait $GPU_KEEPALIVE_PID
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU workload completed."
    rm -f "$KEEPALIVE_LOG"
}

while true; do
    run_keepalive_workload $WORKLOAD_DURATION
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for ${INTERVAL_SECONDS}s until next workload..."
    sleep $INTERVAL_SECONDS
done
