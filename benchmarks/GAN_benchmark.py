#!/usr/bin/env python3
"""
GAN Model Benchmark
Measures inference time and resource consumption for SNRGan model
Generates histograms and performance statistics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os
import time
import logging
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from amc.gan_snr_predictor import SNRGan, GANConfig, SNRDataProcessor


class ResourceMonitor:
    """Monitor CPU, memory, and GPU resources during inference"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get CPU memory usage in MB"""
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in MB"""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            }
        return {'allocated_mb': 0.0, 'reserved_mb': 0.0}
    
    def get_cpu_percent(self) -> float:
        """Get CPU usage percentage"""
        return self.process.cpu_percent(interval=0.01)


class GANBenchmark:
    """Benchmark suite for GAN model"""
    
    def __init__(self, model_path: str, config_path: str, num_iterations: int = 100, use_compile: bool = False, compile_mode: str = "reduce-overhead"):
        self.num_iterations = num_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resource_monitor = ResourceMonitor(str(self.device))
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.compilation_time = 0.0
        
        # Load configuration and model
        logger.info(f"Loading GAN model from {model_path}")
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        
        # Apply torch.compile if requested
        if self.use_compile:
            self._apply_torch_compile()
        
        # Metrics storage
        self.metrics = {
            'inference_times': [],
            'memory_cpu': defaultdict(list),
            'memory_gpu': defaultdict(list),
            'cpu_percent': [],
        }
    
    def _load_config(self, config_path: str) -> GANConfig:
        """Load GAN configuration"""
        try:
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return GANConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return GANConfig()
    
    def _load_model(self, model_path: str) -> SNRGan:
        """Load pre-trained GAN model"""
        config = self.config
        model = SNRGan(config)
        
        if os.path.exists(model_path):
            try:
                # Try with weights_only=False for PyTorch 2.6+ compatibility
                # This allows loading models with numpy objects in state dict
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't have weights_only parameter
                    checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
                    model.generator.load_state_dict(checkpoint['generator_state_dict'])
                    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
                    model.generator.load_state_dict(checkpoint['generator'])
                else:
                    model.generator.load_state_dict(checkpoint)
                logger.info(f"Loaded model: {model_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Using random initialization.")
        else:
            logger.warning(f"Model file not found: {model_path}")
        
        # SNRGan is a wrapper class, not an nn.Module, so move its submodules to device
        model.generator.to(self.device)
        model.discriminator.to(self.device)
        
        # Put generator in evaluation mode (SNRGan.eval() doesn't exist, but generator.eval() does)
        model.generator.eval()
        
        return model
    
    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the generator"""
        try:
            if hasattr(torch, 'compile'):
                logger.info(f"Applying torch.compile with mode '{self.compile_mode}'")
                compile_start = time.perf_counter()
                # Ensure model.generator is a proper PyTorch module before compiling
                if hasattr(self.model, 'generator') and hasattr(self.model.generator, 'forward'):
                    self.model.generator = torch.compile(self.model.generator, mode=self.compile_mode)
                else:
                    logger.warning("Model generator not found or not a proper PyTorch module")
                    self.use_compile = False
                    return
                self.compilation_time = time.perf_counter() - compile_start
                logger.info(f"torch.compile applied successfully (compilation time: {self.compilation_time:.2f}s)")
            else:
                logger.warning("torch.compile not available in this PyTorch version (requires PyTorch 2.0+)")
                self.use_compile = False
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}. Continuing without compilation.")
            self.use_compile = False
    
    def _generate_input_data(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random SNR history input data (history, aux_features, noise)"""
        history_len = self.config.history_len
        aux_dim = self.config.aux_dim
        noise_dim = self.config.noise_dim
        
        # SNRGenerator requires three separate inputs
        snr_history = torch.randn(batch_size, history_len, device=self.device, dtype=self.config.dtype)
        aux_data = torch.randn(batch_size, aux_dim, device=self.device, dtype=self.config.dtype)
        noise = torch.randn(batch_size, noise_dim, device=self.device, dtype=self.config.dtype)
        
        return snr_history, aux_data, noise
    
    def benchmark_inference(self) -> None:
        """Run inference benchmark"""
        logger.info(f"Starting GAN inference benchmark ({self.num_iterations} iterations)")
        
        # Warn if iteration count is very high
        if self.num_iterations > 5000:
            logger.warning(f"⚠️  High iteration count ({self.num_iterations}) may cause GPU memory exhaustion")
            logger.warning("   Consider reducing to 1000-2000 for stable long-term benchmarking")
        
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        time.sleep(0.5)  # Stabilize resources
        
        import gc
        
        with torch.no_grad():
            for i in range(self.num_iterations):
                try:
                    # Get initial memory state
                    mem_before = self.resource_monitor.get_memory_usage()
                    gpu_before = self.resource_monitor.get_gpu_memory_usage()
                    
                    # Time the inference
                    start_time = time.perf_counter()
                    history, aux_features, noise = self._generate_input_data(batch_size=1)
                    output = self.model.generator(history, aux_features, noise)
                    
                    # Synchronize GPU only if not using torch.compile (reduces overhead)
                    # or if compilation completed (every 100 iters for torch.compile runs)
                    if self.device.type == 'cuda' and (not self.use_compile or i % 100 == 0):
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError as e:
                            logger.warning(f"GPU synchronization failed at iteration {i+1}: {e}")
                            # Continue without synchronization to allow benchmark to complete
                            torch.cuda.empty_cache()
                    
                    elapsed_time = time.perf_counter() - start_time
                    
                    # Get final memory state
                    mem_after = self.resource_monitor.get_memory_usage()
                    gpu_after = self.resource_monitor.get_gpu_memory_usage()
                    cpu_percent = self.resource_monitor.get_cpu_percent()
                    
                    # Store metrics
                    self.metrics['inference_times'].append(elapsed_time * 1000)  # Convert to ms
                    
                    for key in mem_before:
                        self.metrics['memory_cpu'][key].append(mem_after[key])
                    
                    for key in gpu_before:
                        self.metrics['memory_gpu'][key].append(gpu_after[key])
                    
                    self.metrics['cpu_percent'].append(cpu_percent)
                    
                    # Periodic GPU cleanup for long-running benchmarks
                    if (i + 1) % 500 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    if (i + 1) % 20 == 0:
                        logger.info(f"Progress: {i+1}/{self.num_iterations} - "
                                  f"Time: {elapsed_time*1000:.2f}ms - "
                                  f"CPU Mem: {mem_after['rss_mb']:.1f}MB")
                
                except Exception as e:
                    logger.error(f"Error at iteration {i+1}: {e}")
                    logger.error("Benchmark terminated due to GPU/resource issues")
                    logger.error("Recommendation: Reduce --iterations to < 5000 and retry")
                    break
        
        logger.info("Inference benchmark completed")
    
    def print_statistics(self) -> Dict:
        """Print and return benchmark statistics"""
        times = np.array(self.metrics['inference_times'])
        
        stats = {
            'total_iterations': self.num_iterations,
            'device': str(self.device),
            'torch_compile': {
                'enabled': self.use_compile,
                'mode': self.compile_mode if self.use_compile else None,
                'compilation_time_sec': float(self.compilation_time) if self.use_compile else 0.0,
            },
            'inference_time_ms': {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'median': float(np.median(times)),
                'p95': float(np.percentile(times, 95)),
                'p99': float(np.percentile(times, 99)),
            },
            'memory_cpu_mb': {
                'rss_mean': float(np.mean(self.metrics['memory_cpu']['rss_mb'])),
                'rss_max': float(np.max(self.metrics['memory_cpu']['rss_mb'])),
                'vms_mean': float(np.mean(self.metrics['memory_cpu']['vms_mb'])),
            },
            'memory_gpu_mb': {
                'allocated_mean': float(np.mean(self.metrics['memory_gpu']['allocated_mb'])),
                'allocated_max': float(np.max(self.metrics['memory_gpu']['allocated_mb'])),
                'reserved_mean': float(np.mean(self.metrics['memory_gpu']['reserved_mb'])),
            },
            'cpu_percent': {
                'mean': float(np.mean(self.metrics['cpu_percent'])),
                'max': float(np.max(self.metrics['cpu_percent'])),
            }
        }
        
        logger.info("\n" + "="*60)
        logger.info("GAN INFERENCE BENCHMARK RESULTS")
        logger.info("="*60)
        logger.info(f"Device: {stats['device']}")
        logger.info(f"Total Iterations: {stats['total_iterations']}")
        logger.info(f"\ntorch.compile: {'Enabled' if stats['torch_compile']['enabled'] else 'Disabled'}")
        if stats['torch_compile']['enabled']:
            logger.info(f"  Mode: {stats['torch_compile']['mode']}")
            logger.info(f"  Compilation Time: {stats['torch_compile']['compilation_time_sec']:.2f}s")
        logger.info("\nInference Time (ms):")
        logger.info(f"  Mean: {stats['inference_time_ms']['mean']:.4f}")
        logger.info(f"  Std: {stats['inference_time_ms']['std']:.4f}")
        logger.info(f"  Min: {stats['inference_time_ms']['min']:.4f}")
        logger.info(f"  Max: {stats['inference_time_ms']['max']:.4f}")
        logger.info(f"  Median: {stats['inference_time_ms']['median']:.4f}")
        logger.info(f"  P95: {stats['inference_time_ms']['p95']:.4f}")
        logger.info(f"  P99: {stats['inference_time_ms']['p99']:.4f}")
        logger.info("\nCPU Memory (MB):")
        logger.info(f"  RSS Mean: {stats['memory_cpu_mb']['rss_mean']:.2f}")
        logger.info(f"  RSS Max: {stats['memory_cpu_mb']['rss_max']:.2f}")
        logger.info("\nGPU Memory (MB):")
        logger.info(f"  Allocated Mean: {stats['memory_gpu_mb']['allocated_mean']:.2f}")
        logger.info(f"  Allocated Max: {stats['memory_gpu_mb']['allocated_max']:.2f}")
        logger.info("\nCPU Usage (%):")
        logger.info(f"  Mean: {stats['cpu_percent']['mean']:.2f}")
        logger.info(f"  Max: {stats['cpu_percent']['max']:.2f}")
        logger.info("="*60 + "\n")
        
        return stats
    
    def plot_histograms(self, output_dir: str = "results") -> None:
        """Generate benchmark histograms"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('GAN Model Benchmark Results', fontsize=16, fontweight='bold')
        
        # Create grid for better layout
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # ===== INFERENCE TIME ANALYSIS =====
        times = np.array(self.metrics['inference_times'])
        warmup_iterations = min(10, len(times) // 10)  # Skip first 10% or 10 iterations, whichever is smaller
        times_stable = times[warmup_iterations:]
        
        # 1. Inference Time Histogram (with warmup)
        ax1 = fig.add_subplot(gs[0, 0])
        # Use Freedman-Diaconis rule for bin size
        iqr = np.percentile(times_stable, 75) - np.percentile(times_stable, 25)
        bin_width = 2 * iqr / (len(times_stable) ** (1/3)) if iqr > 0 else 1
        num_bins = max(10, int((times_stable.max() - times_stable.min()) / bin_width))
        
        ax1.hist(times_stable, bins=num_bins, color='steelblue', edgecolor='black', alpha=0.7, label='Stable runs')
        ax1.axvline(np.mean(times_stable), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(times_stable):.3f}ms')
        ax1.axvline(np.median(times_stable), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(times_stable):.3f}ms')
        ax1.axvline(np.percentile(times_stable, 95), color='orange', linestyle=':', linewidth=2, label=f'P95: {np.percentile(times_stable, 95):.3f}ms')
        ax1.set_xlabel('Inference Time (ms)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'Inference Time Distribution (Stable: iterations {warmup_iterations+1}-{len(times)})', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Log-scale Histogram for full range
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(times, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='All runs')
        ax2.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.3f}ms')
        ax2.set_xlabel('Inference Time (ms)', fontsize=11)
        ax2.set_ylabel('Frequency (log scale)', fontsize=11)
        ax2.set_yscale('log')
        ax2.set_title('Inference Time Distribution (Log Scale, All Iterations)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, which='both')
        
        # 3. Box Plot Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        box_data = [times[:warmup_iterations], times_stable]
        bp = ax3.boxplot(box_data, labels=['Warmup', 'Stable'], patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
            patch.set_facecolor(color)
        ax3.set_ylabel('Inference Time (ms)', fontsize=11)
        ax3.set_title('Warmup vs. Stable Performance Comparison', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Warmup ({warmup_iterations}): {np.mean(times[:warmup_iterations]):.2f}ms\nStable ({len(times_stable)}): {np.mean(times_stable):.2f}ms"
        ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=9, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. CPU Memory Usage (with better scaling)
        ax4 = fig.add_subplot(gs[1, 1])
        rss_mem = np.array(self.metrics['memory_cpu']['rss_mb'])
        mem_variation = rss_mem.max() - rss_mem.min()
        
        if mem_variation > 0.1:  # Show histogram if there's meaningful variation
            ax4.hist(rss_mem, bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax4.axvline(np.mean(rss_mem), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rss_mem):.1f}MB')
            ax4.set_xlabel('Memory Usage (MB)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('CPU Memory Distribution', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(alpha=0.3)
        else:
            # Show summary text if variation is minimal
            ax4.axis('off')
            summary = f"""CPU Memory Usage Summary:
Mean:     {np.mean(rss_mem):>8.2f} MB
Std:      {np.std(rss_mem):>8.4f} MB
Min:      {rss_mem.min():>8.2f} MB
Max:      {rss_mem.max():>8.2f} MB
Variation: {mem_variation:>6.2f} MB

Note: Minimal variation detected"""
            ax4.text(0.5, 0.5, summary, fontsize=10, family='monospace', 
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('CPU Memory Distribution', fontsize=12, fontweight='bold')
        
        # 5. GPU Memory Usage
        ax5 = fig.add_subplot(gs[2, 0])
        gpu_mem = np.array(self.metrics['memory_gpu']['allocated_mb'])
        gpu_variation = gpu_mem.max() - gpu_mem.min()
        
        if gpu_variation > 0.01:  # Show histogram if meaningful variation
            ax5.hist(gpu_mem, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
            ax5.axvline(np.mean(gpu_mem), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gpu_mem):.2f}MB')
            ax5.set_xlabel('GPU Memory Allocated (MB)', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title('GPU Memory Distribution', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(alpha=0.3)
        else:
            # Show summary text if variation is minimal
            ax5.axis('off')
            summary = f"""GPU Memory Usage Summary:
Mean:     {np.mean(gpu_mem):>8.3f} MB
Std:      {np.std(gpu_mem):>8.5f} MB
Min:      {gpu_mem.min():>8.3f} MB
Max:      {gpu_mem.max():>8.3f} MB
Variation: {gpu_variation:>6.3f} MB

Note: GPU memory nearly constant"""
            ax5.text(0.5, 0.5, summary, fontsize=10, family='monospace',
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax5.set_title('GPU Memory Distribution', fontsize=12, fontweight='bold')
        
        # 6. CPU Usage Percentage
        ax6 = fig.add_subplot(gs[2, 1])
        cpu_usage = np.array(self.metrics['cpu_percent'])
        # Use Freedman-Diaconis binning
        cpu_iqr = np.percentile(cpu_usage, 75) - np.percentile(cpu_usage, 25)
        cpu_bin_width = 2 * cpu_iqr / (len(cpu_usage) ** (1/3)) if cpu_iqr > 0 else 5
        cpu_num_bins = max(8, int((cpu_usage.max() - cpu_usage.min()) / cpu_bin_width))
        
        ax6.hist(cpu_usage, bins=cpu_num_bins, color='lightyellow', edgecolor='black', alpha=0.7)
        ax6.axvline(np.mean(cpu_usage), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cpu_usage):.1f}%')
        ax6.axvline(np.median(cpu_usage), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(cpu_usage):.1f}%')
        ax6.set_xlabel('CPU Usage (%)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('CPU Usage Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir) / 'gan_benchmark_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Histograms saved to {output_path}")
        
        # Also create a timeline plot
        self._plot_timeline(output_dir)
    
    def _plot_timeline(self, output_dir: str) -> None:
        """Plot performance metrics over time"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('GAN Model Performance Over Time', fontsize=16, fontweight='bold')
        
        iterations = np.arange(self.num_iterations)
        
        # Inference time timeline
        ax = axes[0]
        ax.plot(iterations, self.metrics['inference_times'], linewidth=1, alpha=0.7, color='steelblue')
        rolling_mean = np.convolve(self.metrics['inference_times'], np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(self.metrics['inference_times'])), rolling_mean, linewidth=2, color='red', label='10-iter MA')
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title('Inference Time Over Iterations', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Memory timeline
        ax = axes[1]
        ax.plot(iterations, self.metrics['memory_cpu']['rss_mb'], linewidth=1, alpha=0.7, color='coral', label='CPU Memory')
        ax.plot(iterations, self.metrics['memory_gpu']['allocated_mb'], linewidth=1, alpha=0.7, color='lightgreen', label='GPU Memory')
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('Memory Usage Over Iterations', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # CPU usage timeline
        ax = axes[2]
        ax.plot(iterations, self.metrics['cpu_percent'], linewidth=1, alpha=0.7, color='lightyellow')
        rolling_mean_cpu = np.convolve(self.metrics['cpu_percent'], np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(self.metrics['cpu_percent'])), rolling_mean_cpu, linewidth=2, color='red', label='10-iter MA')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('CPU Usage (%)', fontsize=11)
        ax.set_title('CPU Usage Over Iterations', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'gan_performance_timeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Timeline plot saved to {output_path}")
    
    def save_statistics(self, output_dir: str = "results") -> None:
        """Save statistics to JSON file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        stats = self.print_statistics()
        
        output_path = Path(output_dir) / 'gan_benchmark_stats.json'
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {output_path}")
    
    def run_complete_benchmark(self, output_dir: str = "results") -> Dict:
        """Run complete benchmark suite"""
        self.benchmark_inference()
        stats = self.print_statistics()
        self.plot_histograms(output_dir)
        self.save_statistics(output_dir)
        return stats


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark GAN SNR predictor model")
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile for optimization')
    parser.add_argument('--compile-mode', choices=['reduce-overhead', 'reduce-peak-memory', 'default'],
                       default='reduce-overhead', help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Paths
    model_path = Path(__file__).parent.parent / "models" / "gan" / "enhanced_snr_gan.pth"
    config_path = Path(__file__).parent.parent / "models" / "gan" / "enhanced_snr_gan_config.json"
    
    # Create benchmark
    benchmark = GANBenchmark(
        model_path=str(model_path),
        config_path=str(config_path),
        num_iterations=args.iterations,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode
    )
    
    # Run complete benchmark
    benchmark.run_complete_benchmark(args.output_dir)


if __name__ == "__main__":
    main()
