# Model Benchmarks

This directory contains comprehensive performance benchmarking scripts for both the GAN and RL Agent models.

## Overview

Two separate benchmark suites:
- **GAN_benchmark.py** - Benchmarks the SNR GAN model inference
- **RL_agent_benchmark.py** - Benchmarks the RL Agent model inference and decision-making

Both benchmarks measure:
- **Inference Time**: Time taken for model forward pass (in milliseconds)
- **CPU Memory**: Resident Set Size (RSS) and Virtual Memory (VMS) usage
- **GPU Memory**: CUDA memory allocation (if available)
- **CPU Usage**: Percentage CPU utilization
- Additional metrics (Q-values for RL, output statistics, etc.)

## Features

### Time Measurements
- Individual iteration times
- Statistical summaries (mean, std, min, max, median, P95, P99)
- Rolling averages for trend analysis
- Timeline visualization showing performance over iterations

### Resource Monitoring
- Real-time CPU and memory tracking per iteration
- GPU memory allocation monitoring
- CPU utilization percentage
- Peak memory usage statistics

### Visualization
- **Histograms**: Distribution of inference times, memory usage, CPU usage
- **Timeline Plots**: Performance metrics over time with moving averages
- **Statistics JSON**: Detailed metrics saved for further analysis

## Usage

### Running GAN Benchmark

```bash
# Basic run
python benchmarks/GAN_benchmark.py

# With torch.compile optimization
python benchmarks/GAN_benchmark.py --use-compile

# With custom options
python benchmarks/GAN_benchmark.py --use-compile --compile-mode reduce-peak-memory --iterations 200

# Without torch.compile (baseline)
python benchmarks/GAN_benchmark.py --iterations 200
```

**Configuration:**
- Number of iterations: 100 (modify with `--iterations`)
- torch.compile: Disabled by default (enable with `--use-compile`)
- Compile mode: 'reduce-overhead' (change with `--compile-mode`)
- Model path: `enhanced_snr_gan.pth`
- Config path: `enhanced_snr_gan_config.json`
- Output directory: `benchmark_results/`

**Expected Outputs:**
- `gan_benchmark_histograms.png` - Distribution histograms
- `gan_performance_timeline.png` - Performance metrics over time
- `gan_benchmark_stats.json` - Detailed statistics including compilation info

### Running RL Agent Benchmark

```bash
# Basic run
python benchmarks/RL_agent_benchmark.py

# With torch.compile optimization
python benchmarks/RL_agent_benchmark.py --use-compile

# With custom options
python benchmarks/RL_agent_benchmark.py --use-compile --compile-mode reduce-peak-memory --iterations 200
```

**Configuration:**
- Number of iterations: 100 (modify with `--iterations`)
- torch.compile: Disabled by default (enable with `--use-compile`)
- Compile mode: 'reduce-overhead' (change with `--compile-mode`)
- Model checkpoint: `rl_amc_model_safeguard1.pth` (optional)
- Output directory: `benchmark_results/`

**Expected Outputs:**
- `rl_agent_benchmark_histograms.png` - Distribution histograms
- `rl_agent_performance_timeline.png` - Performance metrics over time
- `rl_agent_benchmark_stats.json` - Detailed statistics including compilation info

**Special Features:**
- Benchmark includes decision-making process (action selection + epsilon-greedy)
- Q-value distribution analysis
- State normalization through StandardScaler
- torch.compile support for performance optimization

### Run Both Benchmarks

```bash
# Run both with comparison
python benchmarks/compare_benchmarks.py

# Run both with torch.compile
python benchmarks/compare_benchmarks.py --use-compile

# Use existing results without re-running
python benchmarks/compare_benchmarks.py --skip-run

# Detailed options
python benchmarks/compare_benchmarks.py --use-compile --compile-mode reduce-peak-memory --iterations 200
```

### torch.compile Comparison Tool

A dedicated comparison script to easily measure torch.compile benefits:

```bash
# Compare both models with and without torch.compile
python benchmarks/compile_comparison.py

# Compare only GAN model
python benchmarks/compile_comparison.py --model gan

# Compare only RL Agent
python benchmarks/compile_comparison.py --model rl

# With custom iterations and output directory
python benchmarks/compile_comparison.py --model both --iterations 200 --output-dir my_results
```

This script runs each model twice (baseline and compiled) and shows:
- Baseline latency
- Compiled latency  
- Speedup factor
- Compilation time overhead
- Breakeven point (when amortization pays off)

## Output Files

All results are saved in `benchmark_results/` directory:

### GAN Benchmark Outputs
```
benchmark_results/
├── gan_benchmark_histograms.png      # 2x2 histograms of inference metrics
├── gan_performance_timeline.png      # Time-series plots of performance
└── gan_benchmark_stats.json          # JSON file with all statistics + torch.compile info
```

### RL Agent Benchmark Outputs
```
benchmark_results/
├── rl_agent_benchmark_histograms.png # 2x2 histograms of inference metrics
├── rl_agent_performance_timeline.png # Time-series plots of performance
└── rl_agent_benchmark_stats.json     # JSON file with all statistics + torch.compile info
```

### torch.compile Comparison Outputs
When using `compile_comparison.py`, outputs are organized by model and mode:
```
benchmark_results/
├── gan_baseline/                     # Baseline GAN results (no torch.compile)
│   └── gan_*_benchmark_stats.json
├── gan_compiled/                     # Compiled GAN results (with torch.compile)
│   └── gan_*_benchmark_stats.json
├── rl_baseline/                      # Baseline RL results
│   └── rl_*_benchmark_stats.json
└── rl_compiled/                      # Compiled RL results
    └── rl_*_benchmark_stats.json
```

## Interpreting Results

### Inference Time Histogram
- Shows distribution of inference latencies
- Tighter distribution = more consistent performance
- Red line = mean, Green line = median
- Look for multimodal distributions (different performance modes)

### Memory Distribution
- CPU Memory (RSS): Physical memory used by process
- GPU Memory: CUDA memory allocated for model
- Watch for memory growth over time (potential leaks)

### CPU Usage Distribution
- Shows CPU utilization during inference
- Peak usage indicates computational intensity
- Rolling average smooths out spikes

### Timeline Plots
- Identify performance degradation over time
- Spot anomalies or spikes
- 10-iteration moving averages show trends

### Statistics JSON
Open `gan_benchmark_stats.json` or `rl_agent_benchmark_stats.json` to see:
- Percentiles (P95, P99) useful for SLA compliance
- Mean/Std for distribution characterization
- Peak memory usage for capacity planning

## Customization

### Command-Line Arguments

**GAN Benchmark & RL Agent Benchmark:**
```
--iterations N          Number of iterations (default: 100)
--use-compile          Enable torch.compile optimization
--compile-mode MODE    torch.compile mode: reduce-overhead, reduce-peak-memory, default
--output-dir DIR       Output directory (default: benchmark_results/)
```

**Compare Benchmarks:**
```
--output-dir DIR       Output directory (default: benchmark_results/)
--skip-run            Skip running benchmarks, load existing results
--iterations N        Number of iterations (default: 100)
--use-compile         Enable torch.compile optimization
--compile-mode MODE   torch.compile mode
```

**Compile Comparison Tool:**
```
--model MODEL         Model to benchmark: gan, rl, or both (default: both)
--iterations N        Number of iterations per benchmark (default: 100)
--output-dir DIR      Base output directory for results
```

### Modify Number of Iterations

```bash
# Run with 500 iterations for more stable results
python benchmarks/GAN_benchmark.py --iterations 500

# Run comparison with custom iterations
python benchmarks/compare_benchmarks.py --iterations 300
```

### Change Output Directory

```bash
python benchmarks/GAN_benchmark.py --output-dir my_results/gan
python benchmarks/RL_agent_benchmark.py --output-dir my_results/rl
```

### Adjust Resource Monitoring

Modify the `ResourceMonitor` class in either script to track additional metrics:
- GPU temperature
- Network I/O
- Disk I/O
- Process threads

## Requirements

```
torch>=1.9 (PyTorch 2.6+ recommended for improved security)
numpy
matplotlib
psutil
scikit-learn
```

Install with:
```bash
pip install torch numpy matplotlib psutil scikit-learn
```

### PyTorch Version Notes

- **PyTorch 2.6+**: Uses stricter `weights_only=True` by default. Benchmark scripts automatically handle this.
- **PyTorch 2.0-2.5**: Full `torch.compile` support available.
- **PyTorch <2.0**: `torch.compile` features will be disabled automatically.

## torch.compile Optimization

### Overview
PyTorch 2.0+ introduces `torch.compile`, which compiles the model into optimized graphs for faster inference. This feature can provide significant speedups.

### Modes

- **reduce-overhead** (default): Minimizes Python interpreter overhead. Best for small models and short sequences.
- **reduce-peak-memory**: Optimizes for memory usage. Recommended for memory-constrained environments.
- **default**: Standard compilation mode.

### When to Use torch.compile

✅ **Use when:**
- Running inference repeatedly (amortizes compilation cost)
- Model is relatively small and stable (won't change shapes)
- You have PyTorch 2.0+ installed

❌ **Avoid when:**
- Running inference only a few times (compilation overhead)
- Model shapes vary dynamically
- Using older PyTorch versions

### Compilation Overhead

The first time a compiled model runs, there's overhead as the compilation happens. Both benchmark scripts track this in:
- `torch_compile.compilation_time_sec` - One-time compilation time in seconds
- Subsequent iterations show the performance improvement

### Example: Measuring torch.compile Benefits

```bash
# Run baseline (no compilation)
python benchmarks/GAN_benchmark.py --iterations 200

# Run with torch.compile
python benchmarks/GAN_benchmark.py --iterations 200 --use-compile

# Compare results in gan_benchmark_stats.json
# Look at compilation_time_sec and compare inference_time_ms
```

## Notes

- GPU benchmarks require CUDA-capable device
- Memory measurements are per-process
- CPU usage is measured as percentage (0-100%)
- Results may vary based on system load
- Run on idle system for most accurate results
- First iteration typically slower (model initialization/loading)
- torch.compile requires PyTorch 2.0+
- Compilation overhead amortizes over many inferences

## Practical Examples

### Example 1: Quick Baseline Benchmark
```bash
# Quick check of model performance
python benchmarks/GAN_benchmark.py --iterations 50 --output-dir quick_test
```

### Example 2: Production-Ready Benchmark
```bash
# Comprehensive benchmark for production metrics
python benchmarks/GAN_benchmark.py --iterations 500 --output-dir production_metrics
python benchmarks/RL_agent_benchmark.py --iterations 500 --output-dir production_metrics
```

### Example 3: Evaluating torch.compile Benefits
```bash
# Run dedicated comparison to measure torch.compile improvement
python benchmarks/compile_comparison.py --model gan --iterations 200

# Then check the results
cat benchmark_results/gan_baseline/gan_gan_benchmark_stats.json
cat benchmark_results/gan_compiled/gan_gan_benchmark_stats.json
```

### Example 4: Comprehensive Analysis
```bash
# Run everything with torch.compile enabled
python benchmarks/compare_benchmarks.py --use-compile --iterations 200

# Then generate traditional comparison (reuse results)
python benchmarks/compare_benchmarks.py --skip-run
```

### Example 5: Memory-Constrained Environment
```bash
# Use peak-memory optimization mode
python benchmarks/GAN_benchmark.py --use-compile --compile-mode reduce-peak-memory --iterations 100
python benchmarks/RL_agent_benchmark.py --use-compile --compile-mode reduce-peak-memory --iterations 100
```

## Sample Output

When running `compile_comparison.py`, you'll see output like:

```
[2024-04-07 10:15:32,123] INFO - SUMMARY TABLE
[2024-04-07 10:15:32,123] INFO - ----------------------------------------------------------------------------------------------------
[2024-04-07 10:15:32,123] INFO - Model        Baseline (ms)      Compiled (ms)      Speedup      Compilation Time    
[2024-04-07 10:15:32,123] INFO - ----------------------------------------------------------------------------------------------------
[2024-04-07 10:15:32,123] INFO - GAN          2.1523             1.8934             1.14x        2.34s               
[2024-04-07 10:15:32,123] INFO - RL Agent     0.3421             0.2987             1.15x        1.89s               
[2024-04-07 10:15:32,123] INFO - ----------------------------------------------------------------------------------------------------
```

This shows:
- **Speedup of 1.14x** for GAN means torch.compile makes inference 14% faster
- **Compilation time of 2.34s** is a one-time cost
- **Speedup of 1.15x** for RL Agent means similar improvement

## Troubleshooting

**Issue: GPU memory shows 0.0 MB**
- Check if CUDA is available: `torch.cuda.is_available()`
- Verify GPU support with `nvidia-smi`

**Issue: Very high memory usage**
- Reduce batch size if applicable
- Check for memory leaks in data generation
- Monitor with system tools: `htop`, `nvidia-smi dmon`

**Issue: Inconsistent timing results**
- Close other applications
- Increase number of iterations for better averaging
- Use performance mode (if available)

**Issue: torch.compile not found / AttributeError**
- Requires PyTorch 2.0+
- Check your PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Update PyTorch: `pip install --upgrade torch`
- Script will gracefully fall back to non-compiled mode

**Issue: torch.compile compilation fails**
- Some models may not be compatible with torch.compile
- Try different `--compile-mode` options
- Check PyTorch version and CUDA compatibility
- Script logs the error and continues without compilation

**Issue: Slower performance with torch.compile**
- Compilation overhead may exceed inference gains for very few iterations
- Increase iterations (50+) for fair comparison
- Try `--compile-mode reduce-peak-memory` instead
- Some operations may not be optimizable by the compiler

**Issue: "Weights only load failed" error (PyTorch 2.6+)**
- PyTorch 2.6 changed default to `weights_only=True` for security
- Checkpoints with numpy objects need `weights_only=False`
- Fixed in latest benchmark scripts - they now auto-handle this
- If you still see this, ensure you have the latest benchmark files
- The scripts will gracefully fall back to random initialization

## Example JSON Output

```json
{
  "total_iterations": 100,
  "device": "cuda",
  "torch_compile": {
    "enabled": true,
    "mode": "reduce-overhead",
    "compilation_time_sec": 2.34
  },
  "inference_time_ms": {
    "mean": 2.341,
    "std": 0.523,
    "min": 1.892,
    "max": 4.123,
    "median": 2.231,
    "p95": 3.456,
    "p99": 3.891
  },
  "memory_cpu_mb": {
    "rss_mean": 1024.5,
    "rss_max": 1156.3
  },
  "memory_gpu_mb": {
    "allocated_mean": 512.0,
    "allocated_max": 768.0
  },
  "cpu_percent": {
    "mean": 45.2,
    "max": 98.5
  }
}
```

### Interpreting torch.compile Results

- **compilation_time_sec**: One-time cost (only paid once). Amortized over many inferences.
- **inference_time_ms**: Average latency per inference (including any torch.compile overhead reduction).
- **Speedup**: (baseline mean) / (compiled mean) = compilation benefit
  - If speedup > 1.0: torch.compile is faster
  - If speedup < 1.0: Compilation overhead > benefit (try more iterations)
