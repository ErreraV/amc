# 🎯 Quick Start: Real-Time Latency Histogram Dashboard

## Installation (One-time)

```bash
cd /home/errera/tn/pidr-kd-gan/src/amc_/Project

# Install Flask if not already installed
pip install flask flask-cors

# Verify required files exist
ls data/enhanced_snr_preprocessing.pkl  # Should exist
ls models/gan/enhanced_snr_gan.pth      # Optional, has fallback
```

## Start the System

### Option 1: Full System (Recommended)
```bash
python3 src/runners/run_amc_with_analytics.py
```

This will:
1. Start AMC server on port 9001
2. Start Histogram dashboard on port 5000
3. Show real-time metrics as NS-3 simulator sends requests

**Dashboard URL:** http://localhost:5000

---

### Option 2: Test Without Simulator
```bash
# Generate synthetic data and show histograms
python3 test_histograms.py
```

Output shows ASCII histograms + statistics

---

### Option 3: Dashboard Only (if AMC server already running)
```bash
python3 src/servers/dashboard.py
```

---

## What You'll See

### 1️⃣ Real-Time KPI Cards
```
┌─────────────────┬──────────────────┬─────────────────┐
│ Prediction Accuracy                │ Avg Latency     │
│        92.3%                       │     34.2 ms     │
├─────────────────┼──────────────────┼─────────────────┤
│ Sionna Latency  │ Avg Throughput   │ Success Rate    │
│    28.1 ms      │    4.2 Mbps      │     96.2%       │
└─────────────────┴──────────────────┴─────────────────┘
```

### 2️⃣ Three Interactive Histograms

**Decision Latency Distribution**
- Shows total time to make modulation decision
- Bell curve centered at ~35ms
- P95 around 55ms (95% of decisions faster)

**Sionna Latency Distribution** 
- Shows channel simulation time
- Center at ~28ms
- Occasional spikes up to 80ms

**GAN Latency Distribution**
- Shows neural network inference time
- Very narrow distribution at ~6ms
- Only when proactive prediction triggered

### 3️⃣ Latency Statistics Table
```
Decision Latency          Sionna Latency          GAN Latency
─────────────────────    ──────────────────────   ───────────────
Mean: 34.98 ms           Mean: 28.28 ms          Mean: 6.00 ms
P95:  52.57 ms           P95:  46.85 ms          P95:  8.65 ms
P99:  74.99 ms           P99:  62.69 ms          P99:  10.78 ms
Max:  91.30 ms           Max:  74.41 ms          Max:  11.60 ms
```

### 4️⃣ Latency Breakdown
```
Where the time is spent:

Decision Logic (RL/GAN selection): 6.69 ms
├─ RL agent encoding & action: 2-4 ms
├─ Safety validation: 1-2 ms
└─ Response formatting: <0.5 ms

GAN Prediction (when triggered): 6.00 ms
├─ History preparation: 1 ms
├─ Neural forward pass: 4-5 ms
└─ Modulation mapping: <0.5 ms

Sionna Simulation (channel): 28.28 ms
├─ Socket overhead: 1-2 ms
├─ Network latency: 1-2 ms
└─ Actual simulation: 15-25 ms
```

---

## Understanding the Histograms

### Ideal Shape (Bell Curve ✓)
```
         ╱╲
        ╱  ╲
      ╱      ╲
    ╱          ╲
___╱____________╲___
```
- Centered around mean
- Symmetric tails
- Few outliers
- ✓ System working well

### Right-Skewed (Long tail to right ⚠️)
```
    ╱╲
   ╱  ╲
  ╱    ╲╲
╱       ╲╲╲
____═══════___
```
- Most operations fast, some slow
- Occasional bottlenecks
- Check if Sionna is causing delays

### Bimodal (Two peaks ℹ️)
```
  ╱╲      ╱╲
 ╱  ╲    ╱  ╲
╱    ╲╱╲╱    ╲
```
- Two distinct populations
- **Expected!** GAN (fast ~30ms) vs RL (slower ~50ms)
- Shows decision paths working differently

---

## Key Metrics to Monitor

| Metric | Healthy Range | Warning | Critical |
|--------|---------------|---------|----------|
| **Avg Decision Latency** | 25-40ms | 40-80ms | >100ms |
| **Decision P95** | 40-60ms | 60-100ms | >150ms |
| **Sionna Avg** | 20-35ms | 35-70ms | >100ms |
| **Sionna P95** | 35-55ms | 55-100ms | >150ms |
| **GAN Avg** | 4-8ms | 8-15ms | >20ms |
| **Prediction Accuracy** | >85% | 70-85% | <70% |

---

## Troubleshooting

### Dashboard loads but no histograms appear
**Problem:** Not enough data collected yet  
**Solution:** Wait 30-60 seconds for samples to accumulate

### Histograms all empty or very few bars
**Problem:** AMC server running but not receiving requests  
**Solution:** Check that C++ simulator is sending requests to port 9001

### Latency very high (>150ms)
**Problem:** Performance degradation  
**Solution:**
- Check system CPU/memory: `top`
- Check Sionna is responding: `nc -zv 127.0.0.1 9000`
- Check network: `ping 127.0.0.1`

### GAN histogram is empty
**Problem:** Normal - GAN only triggered for proactive predictions  
**Solution:** This is expected. Should be ~50-60% of requests (when using GAN path)

---

## Advanced Usage

### Export Raw Metrics
```python
from performance_analyzer import RealtimeAnalyzer

analyzer = RealtimeAnalyzer()
# ... metrics collected ...

# Get all metrics
metrics_list = list(analyzer.metrics_buffer)

# Save to CSV
import csv
with open('metrics.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['timestamp', 'flow_id', 'decision_latency_ms', ...])
    for m in metrics_list:
        writer.writerow(vars(m))
```

### Custom Latency Analysis
```python
# Get specific histogram
histograms = analyzer.get_latency_histograms()
decision_hist = histograms['decision_latency']['histogram']

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.bar(decision_hist['bin_centers'], decision_hist['counts'])
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.title('Decision Latency Distribution')
plt.show()
```

### Real-time Alerts
```python
# Trigger alert if P95 latency exceeds threshold
while True:
    stats = analyzer.get_latency_histograms()
    p95 = stats['decision_latency']['stats']['p95']
    
    if p95 > 70:  # ms
        print("⚠️  ALERT: Decision latency P95 exceeds 70ms!")
        # Send alert, log, etc.
    
    time.sleep(5)
```

---

## File Locations
```
/home/ehg2004/amc_/Project/
├── performance_analyzer.py       # Histogram collection logic
├── dashboard.py                  # Flask dashboard with charts.js
├── integrated_amc_gan.py         # Updated with metrics collection
├── run_amc_with_analytics.py     # Main runner (recommended)
├── test_histograms.py            # Demo with sample data
└── HISTOGRAM_ANALYSIS.md         # Detailed documentation
```

---

## Next Steps

1. **Start the system:**
   ```bash
   python3 run_amc_with_analytics.py
   ```

2. **Open dashboard:**
   ```
   http://localhost:5000
   ```

3. **Send requests** from C++ simulator or use test data

4. **Monitor** the histograms updating in real-time

5. **Export data** for offline analysis when needed

---

## Performance Tuning Tips

### If Decision Latency is High
1. Check if Sionna is bottleneck (compare latency breakdown)
2. If RL path is slow, check system load
3. Try reducing GAN ensemble samples

### If Sionna Latency is High
1. Reduce channel complexity settings
2. Check if Sionna server is overloaded
3. Add more Sionna server threads

### If GAN Latency is High
1. Move inference to GPU
2. Use quantized model weights
3. Reduce ensemble sample count

---

## Support

For detailed information, see:
- [HISTOGRAM_ANALYSIS.md](HISTOGRAM_ANALYSIS.md) - Full documentation
- [integrated_amc_gan.py](integrated_amc_gan.py) - Implementation
- [performance_analyzer.py](performance_analyzer.py) - Metrics collection

---

**Ready?** Run this to get started:
```bash
cd /home/ehg2004/amc_/Project
python3 run_amc_with_analytics.py
```

Then open your browser to **http://localhost:5000** 🚀
