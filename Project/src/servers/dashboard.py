#!/usr/bin/env python3
"""
Flask dashboard for real-time latency visualization
"""

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import threading
import requests
from ..amc.performance_analyzer import RealtimeAnalyzer

app = Flask(__name__)
CORS(app)

# Global analyzer instance
analyzer_instance = None
analyzer_lock = threading.Lock()


def set_analyzer(analyzer: RealtimeAnalyzer):
    """Set the analyzer instance"""
    global analyzer_instance
    analyzer_instance = analyzer


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/realtime-stats', methods=['GET'])
def get_realtime_stats():
    """Live performance metrics"""
    window = request.args.get('window', 50, type=int)
    try:
        r = requests.get(f'http://127.0.0.1:5001/api/realtime-stats?window={window}', timeout=1.0)
        return (r.content, r.status_code, r.headers.items())
    except Exception:
        if analyzer_instance is None:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        stats = analyzer_instance.get_real_time_stats(window)
        return jsonify(stats)


@app.route('/api/latency-histograms', methods=['GET'])
def get_latency_histograms():
    """Get latency histograms for all latency types"""
    try:
        r = requests.get('http://127.0.0.1:5001/api/latency-histograms', timeout=1.0)
        return (r.content, r.status_code, r.headers.items())
    except Exception:
        if analyzer_instance is None:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        histograms = analyzer_instance.get_latency_histograms()
        return jsonify(histograms)


@app.route('/api/latency-breakdown', methods=['GET'])
def get_latency_breakdown():
    """Get latency breakdown analysis"""
    try:
        r = requests.get('http://127.0.0.1:5001/api/latency-breakdown', timeout=1.0)
        return (r.content, r.status_code, r.headers.items())
    except Exception:
        if analyzer_instance is None:
            return jsonify({'error': 'Analyzer not initialized'}), 503
        breakdown = analyzer_instance.get_latency_breakdown()
        return jsonify(breakdown)


@app.route('/api/flow/<int:flow_id>/stats', methods=['GET'])
def get_flow_stats(flow_id):
    """Per-flow metrics"""
    if analyzer_instance is None:
        return jsonify({'error': 'Analyzer not initialized'}), 503
    stats = analyzer_instance.get_flow_stats(flow_id)
    return jsonify(stats)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'analyzer_ready': analyzer_instance is not None})


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AMC Real-Time Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .model-indicator {
            text-align: center;
            margin-top: -18px;
            margin-bottom: 22px;
            color: white;
            font-size: 1.05em;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-card h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-unit {
            font-size: 0.7em;
            color: #999;
            margin-left: 5px;
        }
        
        .chart-section {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .chart-section h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .charts-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }
        
        .chart-container {
            position: relative;
            height: 350px;
        }
        
        .stats-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 1.2em;
        }
        
        .error {
            background: #fee;
            color: #c00;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AMC Real-Time Performance Dashboard</h1>
        <div class="model-indicator">Active Model: <span id="modelName">waiting for data...</span></div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <h3>Prediction Accuracy</h3>
                <div class="metric-value"><span id="accuracy">-</span><span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <h3>Decision Latency (Avg)</h3>
                <div class="metric-value"><span id="avgLatency">-</span><span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <h3>Sionna Latency (Avg)</h3>
                <div class="metric-value"><span id="sionnaLatency">-</span><span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <h3>Avg Throughput</h3>
                <div class="metric-value"><span id="throughput">-</span><span class="metric-unit">Mbps</span></div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value"><span id="success">-</span><span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <h3>Avg BLER</h3>
                <div class="metric-value"><span id="bler">-</span><span class="metric-unit"></span></div>
            </div>
        </div>
        
        <!-- Latency Histograms -->
        <div class="chart-section">
            <h2>⏱️ Latency Distribution Histograms</h2>
            <div class="charts-row">
                <div class="chart-container">
                    <canvas id="decisionLatencyHist"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="sionnaLatencyHist"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="ganLatencyHist"></canvas>
                </div>
            </div>
            
            <div class="stats-breakdown" id="latencyStats"></div>
        </div>
        
        <!-- Latency Breakdown -->
        <div class="chart-section">
            <h2>🔍 Latency Components Breakdown</h2>
            <div class="stats-breakdown" id="breakdown"></div>
        </div>
    </div>
    
    <script>
        // Chart.js instances
        const charts = {};
        
        // Color scheme
        const colors = {
            decision: 'rgb(102, 126, 234)',
            sionna: 'rgb(118, 75, 162)',
            gan: 'rgb(220, 100, 180)',
            success: 'rgb(75, 192, 75)'
        };
        
        function initHistogramChart(canvasId, title, color) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: title,
                        data: [],
                        backgroundColor: color,
                        borderColor: color,
                        borderWidth: 1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: { font: { size: 12 } }
                        },
                        title: {
                            display: true,
                            text: title,
                            font: { size: 14, weight: 'bold' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'Count' }
                        },
                        x: {
                            title: { display: true, text: 'Latency (ms)' }
                        }
                    }
                }
            });
        }
        
        // Initialize histogram charts
        charts.decision = initHistogramChart('decisionLatencyHist', 'Decision Latency Distribution', colors.decision);
        charts.sionna = initHistogramChart('sionnaLatencyHist', 'Sionna Latency Distribution', colors.sionna);
        charts.gan = initHistogramChart('ganLatencyHist', 'GAN Latency Distribution', colors.gan);
        
        async function updateDashboard() {
            try {
                // Fetch real-time stats
                const statsRes = await fetch('/api/realtime-stats?window=50');
                const stats = await statsRes.json();
                
                // Fetch histograms
                const histRes = await fetch('/api/latency-histograms');
                const histograms = await histRes.json();
                
                // Fetch breakdown
                const breakdownRes = await fetch('/api/latency-breakdown');
                const breakdown = await breakdownRes.json();
                
                // Update KPI cards
                document.getElementById('accuracy').textContent = 
                    (stats.prediction_accuracy * 100).toFixed(1);
                document.getElementById('avgLatency').textContent = 
                    stats.avg_latency_ms.toFixed(2);
                document.getElementById('sionnaLatency').textContent =
                    'N/A'; // Will be from histogram stats
                document.getElementById('throughput').textContent = 
                    stats.avg_throughput.toFixed(2);
                document.getElementById('success').textContent = 
                    (stats.success_rate * 100).toFixed(1);
                document.getElementById('bler').textContent = 
                    stats.avg_bler.toFixed(4);

                if (stats.current_model && stats.current_model.name) {
                    document.getElementById('modelName').textContent = stats.current_model.name;
                }
                
                // Update decision latency histogram
                if (histograms.decision_latency) {
                    const declat = histograms.decision_latency.histogram;
                    charts.decision.data.labels = declat.bin_centers.map(x => x.toFixed(1));
                    charts.decision.data.datasets[0].data = declat.counts;
                    charts.decision.update();
                    
                    // Update Sionna latency display
                    document.getElementById('sionnaLatency').textContent = 
                        histograms.sionna_latency.stats.mean.toFixed(2);
                }
                
                // Update sionna latency histogram
                if (histograms.sionna_latency) {
                    const sionlat = histograms.sionna_latency.histogram;
                    charts.sionna.data.labels = sionlat.bin_centers.map(x => x.toFixed(1));
                    charts.sionna.data.datasets[0].data = sionlat.counts;
                    charts.sionna.update();
                }
                
                // Update GAN latency histogram
                if (histograms.gan_latency && histograms.gan_latency.stats.count > 0) {
                    const ganlat = histograms.gan_latency.histogram;
                    charts.gan.data.labels = ganlat.bin_centers.map(x => x.toFixed(1));
                    charts.gan.data.datasets[0].data = ganlat.counts;
                    charts.gan.update();
                }
                
                // Update latency statistics cards
                let latencyHTML = '';
                for (const [key, data] of Object.entries(histograms)) {
                    const stats_data = data.stats;
                    latencyHTML += `
                        <div class="stat-item">
                            <div class="stat-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                            <div>Mean: <strong>${stats_data.mean.toFixed(2)}ms</strong></div>
                            <div style="font-size: 0.9em; color: #999;">
                                P95: ${stats_data.p95.toFixed(2)}ms | P99: ${stats_data.p99.toFixed(2)}ms | Max: ${stats_data.max.toFixed(2)}ms
                            </div>
                        </div>
                    `;
                }
                document.getElementById('latencyStats').innerHTML = latencyHTML;
                
                // Update breakdown
                let breakdownHTML = '';
                for (const [component, data] of Object.entries(breakdown)) {
                    breakdownHTML += `
                        <div class="stat-item">
                            <div class="stat-label">${component.replace(/_/g, ' ').toUpperCase()}</div>
                            <div>Mean: <strong>${data.mean_ms.toFixed(2)}ms</strong></div>
                            <div style="font-size: 0.9em; color: #999;">
                                ±${data.std_ms.toFixed(2)}ms | Max: ${data.max_ms.toFixed(2)}ms
                            </div>
                        </div>
                    `;
                }
                document.getElementById('breakdown').innerHTML = breakdownHTML;
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        // Update every 1 second
        setInterval(updateDashboard, 1000);
        updateDashboard(); // Initial load
    </script>
</body>
</html>
'''


def run_dashboard(analyzer: RealtimeAnalyzer, host='0.0.0.0', port=5000, debug=False):
    """Start the dashboard server"""
    set_analyzer(analyzer)
    print(f"Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    from performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
    import numpy as np
    
    # Create test data
    analyzer = RealtimeAnalyzer()
    
    for i in range(200):
        metrics = PerformanceMetrics(
            flow_id=i % 10,
            decision_method='preselected_from_gan_prediction' if i % 2 == 0 else 'rl_agent_current_snr',
            predicted_snr=20 + np.random.randn() * 2,
            actual_snr=20 + np.random.randn() * 2,
            prediction_error=abs(np.random.randn() * 3),
            selected_modulation=np.random.choice(['qam4', 'qam16', 'qam64', 'qam256']),
            ber=np.random.exponential(1e-5),
            bler=np.random.uniform(0, 0.3),
            throughput=np.random.uniform(0, 8),
            decision_latency_ms=np.abs(np.random.normal(35, 12)),
            sionna_latency_ms=np.abs(np.random.normal(28, 10)),
            gan_latency_ms=np.abs(np.random.normal(6, 2.5)) if i % 2 == 0 else 0,
            success=np.random.random() > 0.1
        )
        analyzer.record_decision(metrics)
    
    run_dashboard(analyzer, debug=True)
