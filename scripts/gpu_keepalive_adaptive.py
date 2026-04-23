#!/usr/bin/env python3
"""
Adaptive GPU Keepalive Script
Monitors GPU utilization and performs matrix multiplications when it drops below threshold.
"""

import argparse
import signal
import sys
import time
import torch

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")
    print("Running in continuous mode without utilization monitoring.")


class GPUKeepalive:
    def __init__(self, device_id=0, threshold=50, check_interval=5, matrix_size=4096):
        self.device_id = device_id
        self.threshold = threshold
        self.check_interval = check_interval
        self.matrix_size = matrix_size
        self.running = True
        self.device = f'cuda:{device_id}'
        
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        if NVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        self.tensor = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.bfloat16)
        print(f"GPU keepalive initialized on {self.device}", flush=True)
        print(f"Threshold: {threshold}%, Check interval: {check_interval}s, Matrix size: {matrix_size}x{matrix_size}", flush=True)
    
    def _signal_handler(self, signum, frame):
        print("\nShutting down GPU keepalive...", flush=True)
        self.running = False
        if NVML_AVAILABLE:
            pynvml.nvmlShutdown()
        sys.exit(0)
    
    def get_gpu_utilization(self):
        """Get current GPU utilization percentage."""
        if not NVML_AVAILABLE:
            return 0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except Exception as e:
            print(f"Error reading GPU utilization: {e}", flush=True)
            return 0
    
    def run_continuous(self):
        """Run matrix multiplications continuously."""
        print("Running in continuous mode...", flush=True)
        while self.running:
            torch.matmul(self.tensor, self.tensor)
            time.sleep(0.0005)
    
    def run_adaptive(self):
        """Run matrix multiplications only when utilization is below threshold."""
        print("Running in adaptive mode...", flush=True)
        
        while self.running:
            utilization = self.get_gpu_utilization()
            
            if utilization < self.threshold:
                print(f"GPU utilization: {utilization}% (below {self.threshold}%) - Starting keepalive workload", flush=True)
                
                start_time = time.time()
                while self.running and (time.time() - start_time) < self.check_interval:
                    torch.matmul(self.tensor, self.tensor)
                    time.sleep(0.0005)
            else:
                print(f"GPU utilization: {utilization}% (above {self.threshold}%) - Idle", flush=True)
                time.sleep(self.check_interval)
    
    def run(self):
        """Run keepalive in appropriate mode."""
        if NVML_AVAILABLE:
            self.run_adaptive()
        else:
            self.run_continuous()


def main():
    parser = argparse.ArgumentParser(description='GPU Keepalive Script')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--threshold', type=int, default=50, help='Utilization threshold percentage (default: 50)')
    parser.add_argument('--interval', type=int, default=5, help='Check interval in seconds (default: 5)')
    parser.add_argument('--matrix-size', type=int, default=4096, help='Matrix size for matmul (default: 4096)')
    parser.add_argument('--continuous', action='store_true', help='Run continuously without monitoring')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available", file=sys.stderr)
        sys.exit(1)
    
    if args.device >= torch.cuda.device_count():
        print(f"Error: Device cuda:{args.device} not available", file=sys.stderr)
        sys.exit(1)
    
    keepalive = GPUKeepalive(
        device_id=args.device,
        threshold=args.threshold,
        check_interval=args.interval,
        matrix_size=args.matrix_size
    )
    
    if args.continuous or not NVML_AVAILABLE:
        keepalive.run_continuous()
    else:
        keepalive.run_adaptive()


if __name__ == '__main__':
    main()
