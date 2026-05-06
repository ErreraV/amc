#!/usr/bin/env python3
"""
RL Agent Model Benchmark
Measures inference time and resource consumption for AdvancedQLearningAgent model
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from amc_server import AdvancedQLearningAgent
from sklearn.preprocessing import StandardScaler


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


class RLAgentBenchmark:
    """Benchmark suite for RL Agent model"""
    
    def __init__(self, checkpoint_path: str = None, num_iterations: int = 100, use_compile: bool = False, compile_mode: str = "reduce-overhead"):
        self.num_iterations = num_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resource_monitor = ResourceMonitor(str(self.device))
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.compilation_time = 0.0
        
        # Initialize RL agent
        logger.info("Initializing RL Agent")
        self.agent = AdvancedQLearningAgent(
            state_dim=7,
            action_dim=4,
            learning_rate=0.001
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading RL Agent checkpoint from {checkpoint_path}")
            try:
                # Try with weights_only=False for PyTorch 2.6+ compatibility
                # This allows loading models with numpy objects in state dict
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't have weights_only parameter
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                    self.agent.q_network.load_state_dict(checkpoint['q_network'])
                    self.agent.target_network.load_state_dict(checkpoint['target_network'])
                else:
                    self.agent.q_network.load_state_dict(checkpoint)
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Using random initialization.")
        else:
            logger.info(f"No checkpoint provided or found. Using random initialization.")
        
        self.agent.q_network.to(self.device)
        self.agent.target_network.to(self.device)
        self.agent.q_network.eval()
        self.agent.target_network.eval()
        
        # Apply torch.compile if requested
        if self.use_compile:
            self._apply_torch_compile()
        
        # Metrics storage
        self.metrics = {
            'inference_times': [],
            'memory_cpu': defaultdict(list),
            'memory_gpu': defaultdict(list),
            'cpu_percent': [],
            'q_values': [],
        }
    
    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the Q-network"""
        try:
            if hasattr(torch, 'compile'):
                logger.info(f"Applying torch.compile with mode '{self.compile_mode}'")
                compile_start = time.perf_counter()
                self.agent.q_network = torch.compile(self.agent.q_network, mode=self.compile_mode)
                self.compilation_time = time.perf_counter() - compile_start
                logger.info(f"torch.compile applied successfully (compilation time: {self.compilation_time:.2f}s)")
            else:
                logger.warning("torch.compile not available in this PyTorch version (requires PyTorch 2.0+)")
                self.use_compile = False
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}. Continuing without compilation.")
            self.use_compile = False
    
    def _generate_state(self) -> np.ndarray:
        """Generate random state for RL agent inference"""
        state_dim = self.agent.state_dim
        state = np.random.randn(state_dim).astype(np.float32)
        
        # Normalize using StandardScaler
        if not hasattr(self.agent.state_scaler, 'mean_'):
            self.agent.state_scaler.fit([state])
        
        state_normalized = self.agent.state_scaler.transform([state])[0]
        return state_normalized
    
    def benchmark_inference(self) -> None:
        """Run inference benchmark"""
        logger.info(f"Starting RL Agent inference benchmark ({self.num_iterations} iterations)")
        
        # Warn if iteration count is very high
        if self.num_iterations > 5000:
            logger.warning(f"⚠️  High iteration count ({self.num_iterations}) may cause GPU memory exhaustion")
            logger.warning("   Consider reducing to 1000-2000 for stable long-term benchmarking")
        
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        time.sleep(0.5)  # Stabilize resources
        
        with torch.no_grad():
            for i in range(self.num_iterations):
                # Get initial memory state
                mem_before = self.resource_monitor.get_memory_usage()
                gpu_before = self.resource_monitor.get_gpu_memory_usage()
                
                # Generate random state
                state = self._generate_state()
                state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                
                # Time the inference (Q-network forward pass)
                start_time = time.perf_counter()
                q_values = self.agent.q_network(state_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.perf_counter() - start_time
                
                # Get final memory state
                mem_after = self.resource_monitor.get_memory_usage()
                gpu_after = self.resource_monitor.get_gpu_memory_usage()
                cpu_percent = self.resource_monitor.get_cpu_percent()
                
                # Store metrics
                self.metrics['inference_times'].append(elapsed_time * 1000)  # Convert to ms
                self.metrics['q_values'].append(q_values.detach().cpu().numpy().flatten()[0])
                
                for key in mem_before:
                    self.metrics['memory_cpu'][key].append(mem_after[key])
                
                for key in gpu_before:
                    self.metrics['memory_gpu'][key].append(gpu_after[key])
                
                self.metrics['cpu_percent'].append(cpu_percent)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Progress: {i+1}/{self.num_iterations} - "
                              f"Time: {elapsed_time*1000:.2f}ms - "
                              f"CPU Mem: {mem_after['rss_mb']:.1f}MB - "
                              f"Q-value: {q_values[0, 0]:.4f}")
        
        logger.info("Inference benchmark completed")
    
    def benchmark_decision_making(self) -> None:
        """Benchmark complete decision-making process including action selection"""
        logger.info(f"Starting RL Agent decision-making benchmark ({self.num_iterations} iterations)")
        
        # Warn if iteration count is very high
        if self.num_iterations > 5000:
            logger.warning(f"⚠️  High iteration count ({self.num_iterations}) may cause GPU memory exhaustion")
            logger.warning("   Consider reducing to 1000-2000 for stable long-term benchmarking")
        
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        time.sleep(0.5)
        
        decision_times = []
        
        with torch.no_grad():
            for i in range(self.num_iterations):
                state = self._generate_state()
                state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                
                # Time the complete decision-making process
                start_time = time.perf_counter()
                
                # Forward pass through Q-network
                q_values = self.agent.q_network(state_tensor)
                
                # Epsilon-greedy action selection
                if np.random.random() < self.agent.epsilon:
                    action = np.random.randint(0, self.agent.action_dim)
                else:
                    action = torch.argmax(q_values, dim=1).item()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.perf_counter() - start_time
                decision_times.append(elapsed_time * 1000)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Progress: {i+1}/{self.num_iterations} - "
                              f"Decision Time: {elapsed_time*1000:.4f}ms - "
                              f"Selected Action: {action}")
        
        # Add to metrics
        self.metrics['decision_times'] = decision_times
        logger.info("Decision-making benchmark completed")
    
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
            },
            'q_values': {
                'mean': float(np.mean(self.metrics['q_values'])),
                'std': float(np.std(self.metrics['q_values'])),
                'min': float(np.min(self.metrics['q_values'])),
                'max': float(np.max(self.metrics['q_values'])),
            }
        }
        
        # Add decision-making stats if available
        if 'decision_times' in self.metrics:
            decision_times = np.array(self.metrics['decision_times'])
            stats['decision_time_ms'] = {
                'mean': float(np.mean(decision_times)),
                'std': float(np.std(decision_times)),
                'min': float(np.min(decision_times)),
                'max': float(np.max(decision_times)),
                'median': float(np.median(decision_times)),
            }
        
        logger.info("\n" + "="*60)
        logger.info("RL AGENT INFERENCE BENCHMARK RESULTS")
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
        
        if 'decision_time_ms' in stats:
            logger.info("\nDecision-Making Time (ms):")
            logger.info(f"  Mean: {stats['decision_time_ms']['mean']:.4f}")
            logger.info(f"  Std: {stats['decision_time_ms']['std']:.4f}")
            logger.info(f"  Min: {stats['decision_time_ms']['min']:.4f}")
            logger.info(f"  Max: {stats['decision_time_ms']['max']:.4f}")
            logger.info(f"  Median: {stats['decision_time_ms']['median']:.4f}")
        
        logger.info("\nCPU Memory (MB):")
        logger.info(f"  RSS Mean: {stats['memory_cpu_mb']['rss_mean']:.2f}")
        logger.info(f"  RSS Max: {stats['memory_cpu_mb']['rss_max']:.2f}")
        logger.info("\nGPU Memory (MB):")
        logger.info(f"  Allocated Mean: {stats['memory_gpu_mb']['allocated_mean']:.2f}")
        logger.info(f"  Allocated Max: {stats['memory_gpu_mb']['allocated_max']:.2f}")
        logger.info("\nCPU Usage (%):")
        logger.info(f"  Mean: {stats['cpu_percent']['mean']:.2f}")
        logger.info(f"  Max: {stats['cpu_percent']['max']:.2f}")
        logger.info("\nQ-Values Statistics:")
        logger.info(f"  Mean: {stats['q_values']['mean']:.4f}")
        logger.info(f"  Std: {stats['q_values']['std']:.4f}")
        logger.info(f"  Range: [{stats['q_values']['min']:.4f}, {stats['q_values']['max']:.4f}]")
        logger.info("="*60 + "\n")
        
        return stats
    
    def plot_histograms(self, output_dir: str = "benchmark_results") -> None:
        """Generate benchmark histograms"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('RL Agent Model Benchmark Results', fontsize=16, fontweight='bold')
        
        # Create grid for better layout
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # ===== INFERENCE TIME ANALYSIS =====
        times = np.array(self.metrics['inference_times'])
        warmup_iterations = min(10, len(times) // 10)  # Skip first 10% or 10 iterations, whichever is smaller
        times_stable = times[warmup_iterations:] if len(times) > warmup_iterations else times
        
        # 1. Inference Time Histogram (with warmup)
        ax1 = fig.add_subplot(gs[0, 0])
        # Use Freedman-Diaconis rule for bin size
        if len(times_stable) > 1:
            iqr = np.percentile(times_stable, 75) - np.percentile(times_stable, 25)
            bin_width = 2 * iqr / (len(times_stable) ** (1/3)) if iqr > 0 else 1
            num_bins = max(10, int((times_stable.max() - times_stable.min()) / bin_width)) if bin_width > 0 else 30
        else:
            num_bins = 10
            
        ax1.hist(times_stable, bins=num_bins, color='steelblue', edgecolor='black', alpha=0.7, label='Stable runs')
        ax1.axvline(np.mean(times_stable), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(times_stable):.3f}ms')
        ax1.axvline(np.median(times_stable), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(times_stable):.3f}ms')
        if len(times_stable) > 0:
            p95 = np.percentile(times_stable, 95)
            ax1.axvline(p95, color='orange', linestyle=':', linewidth=2, label=f'P95: {p95:.3f}ms')
            
            # Adaptive x-axis limiting
            p99 = np.percentile(times_stable, 99)
            xlim_max = min(p99 * 1.5, times_stable.max())
            outliers_beyond = np.sum(times_stable > xlim_max)
            
            ax1.set_xlim(left=0, right=xlim_max)
            if outliers_beyond > 0:
                outlier_pct = 100 * outliers_beyond / len(times_stable)
                ax1.text(0.98, 0.97, f'{outliers_beyond} outliers beyond axis\n({outlier_pct:.1f}%)', 
                        transform=ax1.transAxes, fontsize=8, verticalalignment='top', 
                        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
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
        box_data = [times[:warmup_iterations], times_stable] if warmup_iterations > 0 else [times_stable]
        labels = ['Warmup', 'Stable'] if warmup_iterations > 0 else ['Stable']
        bp = ax3.boxplot(box_data, patch_artist=True, widths=0.5)
        ax3.set_xticks(range(1, len(labels) + 1))
        ax3.set_xticklabels(labels)
        colors = ['lightcoral', 'lightgreen'] if warmup_iterations > 0 else ['lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax3.set_ylabel('Inference Time (ms)', fontsize=11)
        ax3.set_title('Warmup vs. Stable Performance Comparison', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
        
        # Add statistics text
        if warmup_iterations > 0:
            stats_text = f"Warmup ({warmup_iterations}): {np.mean(times[:warmup_iterations]):.2f}ms\nStable ({len(times_stable)}): {np.mean(times_stable):.2f}ms"
        else:
            stats_text = f"Stable ({len(times_stable)}): {np.mean(times_stable):.2f}ms"
        ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=9, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. CPU Memory Usage
        ax4 = fig.add_subplot(gs[1, 1])
        rss_mem = np.array(self.metrics['memory_cpu']['rss_mb'])
        mem_variation = rss_mem.max() - rss_mem.min() if len(rss_mem) > 0 else 0
        
        if mem_variation > 0.1:
            ax4.hist(rss_mem, bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax4.axvline(np.mean(rss_mem), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rss_mem):.1f}MB')
            ax4.set_xlabel('Memory Usage (MB)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('CPU Memory Distribution', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(alpha=0.3)
        else:
            ax4.axis('off')
            if len(rss_mem) > 0:
                summary = f"CPU Memory Usage Summary:\nMean:     {np.mean(rss_mem):>8.2f} MB\nStd:      {np.std(rss_mem):>8.4f} MB\nMin:      {rss_mem.min():>8.2f} MB\nMax:      {rss_mem.max():>8.2f} MB\nVariation: {mem_variation:>6.2f} MB\n\nNote: Minimal variation detected"
                ax4.text(0.5, 0.5, summary, fontsize=10, family='monospace', verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax4.set_title('CPU Memory Distribution', fontsize=12, fontweight='bold')
        
        # 5. Q-Values Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        q_vals = np.array(self.metrics['q_values'])
        q_variation = q_vals.max() - q_vals.min() if len(q_vals) > 0 else 0
        
        if q_variation > 0.001:
            if len(q_vals) > 1:
                q_iqr = np.percentile(q_vals, 75) - np.percentile(q_vals, 25)
                q_bin_width = 2 * q_iqr / (len(q_vals) ** (1/3)) if q_iqr > 0 else 1
                q_num_bins = max(10, int((q_vals.max() - q_vals.min()) / q_bin_width)) if q_bin_width > 0 else 30
            else:
                q_num_bins = 10
            ax5.hist(q_vals, bins=q_num_bins, color='lightgreen', edgecolor='black', alpha=0.7)
            ax5.axvline(np.mean(q_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(q_vals):.4f}')
            ax5.set_xlabel('Q-Value', fontsize=11)
            ax5.set_ylabel('Frequency', fontsize=11)
            ax5.set_title('Q-Values Distribution', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(alpha=0.3)
        else:
            ax5.axis('off')
            if len(q_vals) > 0:
                summary = f"Q-Values Summary:\nMean:     {np.mean(q_vals):>8.4f}\nStd:      {np.std(q_vals):>8.4f}\nMin:      {q_vals.min():>8.4f}\nMax:      {q_vals.max():>8.4f}\nVariation: {q_variation:>6.4f}\n\nNote: Minimal variation detected"
                ax5.text(0.5, 0.5, summary, fontsize=10, family='monospace', verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax5.set_title('Q-Values Distribution', fontsize=12, fontweight='bold')

        # 6. CPU Usage Percentage
        ax6 = fig.add_subplot(gs[2, 1])
        cpu_usage = np.array(self.metrics['cpu_percent'])
        if len(cpu_usage) > 1:
            cpu_iqr = np.percentile(cpu_usage, 75) - np.percentile(cpu_usage, 25)
            cpu_bin_width = 2 * cpu_iqr / (len(cpu_usage) ** (1/3)) if cpu_iqr > 0 else 5
            cpu_num_bins = max(8, int((cpu_usage.max() - cpu_usage.min()) / cpu_bin_width)) if cpu_bin_width > 0 else 30
        else:
            cpu_num_bins = 10
            
        ax6.hist(cpu_usage, bins=cpu_num_bins, color='lightyellow', edgecolor='black', alpha=0.7)
        ax6.axvline(np.mean(cpu_usage), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cpu_usage):.2f}%')
        ax6.axvline(np.median(cpu_usage), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(cpu_usage):.2f}%')
        ax6.set_xlabel('CPU Usage (%)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('CPU Usage Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir) / 'rl_agent_benchmark_histograms.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Histograms saved to {output_path}")
        
        # Also create a timeline plot
        self._plot_timeline(output_dir)
    
    def _plot_timeline(self, output_dir: str) -> None:
        """Plot performance metrics over time"""
        try:
            # Use actual data length, not expected num_iterations (in case of early termination)
            actual_iterations = len(self.metrics['inference_times'])
            
            if actual_iterations < 2:
                logger.warning("Not enough data points to plot timeline (minimum 2 required)")
                return
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle('RL Agent Model Performance Over Time', fontsize=16, fontweight='bold')
            
            iterations = np.arange(actual_iterations)
            
            # Inference time timeline
            ax = axes[0]
            ax.plot(iterations, self.metrics['inference_times'], linewidth=1, alpha=0.7, color='steelblue')
            if actual_iterations > 10:
                rolling_mean = np.convolve(self.metrics['inference_times'], np.ones(10)/10, mode='valid')
                ax.plot(range(9, len(self.metrics['inference_times'])), rolling_mean, linewidth=2, color='red', label='10-iter MA')
            ax.set_ylabel('Time (ms)', fontsize=11)
            ax.set_title('Inference Time Over Iterations', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Memory and Q-value timeline
            ax = axes[1]
            cpu_mem = np.array(self.metrics['memory_cpu']['rss_mb'])[:actual_iterations]
            q_values = np.array(self.metrics['q_values'])[:actual_iterations]
            ax.plot(iterations, cpu_mem, linewidth=1, alpha=0.7, color='coral', label='CPU Memory')
            ax2 = ax.twinx()
            ax2.plot(iterations, q_values, linewidth=1, alpha=0.7, color='lightgreen', label='Q-Value')
            ax.set_ylabel('Memory (MB)', fontsize=11, color='coral')
            ax2.set_ylabel('Q-Value', fontsize=11, color='lightgreen')
            ax.set_title('Memory and Q-Values Over Iterations', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # CPU usage timeline
            ax = axes[2]
            cpu_usage = np.array(self.metrics['cpu_percent'])[:actual_iterations]
            ax.plot(iterations, cpu_usage, linewidth=1, alpha=0.7, color='lightyellow')
            if actual_iterations > 10:
                rolling_mean_cpu = np.convolve(cpu_usage, np.ones(10)/10, mode='valid')
                ax.plot(range(9, len(cpu_usage)), rolling_mean_cpu, linewidth=2, color='red', label='10-iter MA')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('CPU Usage (%)', fontsize=11)
            ax.set_title('CPU Usage Over Iterations', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            output_path = Path(output_dir) / 'rl_agent_performance_timeline.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Timeline plot saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not generate timeline plot: {e}")
        logger.info(f"Timeline plot saved to {output_path}")
    
    def save_statistics(self, output_dir: str = "benchmark_results") -> None:
        """Save statistics to JSON file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        stats = self.print_statistics()
        
        output_path = Path(output_dir) / 'rl_agent_benchmark_stats.json'
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {output_path}")
    
    def run_complete_benchmark(self, output_dir: str = "benchmark_results") -> Dict:
        """Run complete benchmark suite including decision-making"""
        self.benchmark_inference()
        self.benchmark_decision_making()
        stats = self.print_statistics()
        self.plot_histograms(output_dir)
        self.save_statistics(output_dir)
        return stats


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RL Agent Q-learning model")
    parser.add_argument('--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile for optimization')
    parser.add_argument('--compile-mode', choices=['reduce-overhead', 'reduce-peak-memory', 'default'],
                       default='reduce-overhead', help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Path to checkpoint (optional)
    checkpoint_path = Path(__file__).parent.parent / "rl_amc_model_safeguard1.pth"
    
    # Create benchmark
    benchmark = RLAgentBenchmark(
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        num_iterations=args.iterations,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode
    )
    
    # Run complete benchmark
    benchmark.run_complete_benchmark(args.output_dir)


if __name__ == "__main__":
    main()
