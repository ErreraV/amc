#!/usr/bin/env python3
"""
Flask dashboard for real-time latency visualization.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from ..amc.performance_analyzer import (
    LatencyHistogram,
    summarize_latency_breakdown,
    summarize_latency_histograms,
    summarize_realtime_stats,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DEFAULT_METRICS_SERVER_URL = 'http://127.0.0.1:5001'
app.config['METRICS_SERVER_URL'] = DEFAULT_METRICS_SERVER_URL


def set_analyzer(analyzer):
    """Compatibility shim for older callers; dashboard now reads raw events directly."""
    app.config['ANALYZER'] = analyzer
    logger.info("[Dashboard] Legacy analyzer injection ignored; raw metrics are used instead")

def set_metrics_server_url(url: str):
    """Set the metrics server base URL used by dashboard endpoints."""
    normalized_url = url.rstrip('/')
    app.config['METRICS_SERVER_URL'] = normalized_url
    logger.info(f"[Dashboard] Metrics server URL set to: {normalized_url}")


def _metrics_server_url() -> str:
    return app.config.get('METRICS_SERVER_URL', DEFAULT_METRICS_SERVER_URL).rstrip('/')


def _fetch_recent_events(window: int = 500) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{_metrics_server_url()}/api/events",
        params={'window': window},
        timeout=2,
    )
    response.raise_for_status()
    events = response.json()
    if not isinstance(events, list):
        raise ValueError('Metrics server returned an invalid events payload')
    return events


def _empty_realtime_stats() -> Dict[str, Any]:
    return {
        'window_size': 0,
        'avg_prediction_error': 0.0,
        'prediction_accuracy': 0.0,
        'avg_latency_ms': 0.0,
        'avg_throughput': 0.0,
        'avg_bler': 0.0,
        'avg_ber': 0.0,
        'success_rate': 0.0,
        'last_sample_time': 0.0,
        'decision_method_distribution': {
            'gan_preselected': 0,
            'rl_fallback': 0,
        },
    }


def _empty_latency_histograms() -> Dict[str, Any]:
    return {
        'decision_latency': {
            'histogram': LatencyHistogram(min_ms=0, max_ms=100, num_bins=50).to_dict(),
            'stats': LatencyHistogram(min_ms=0, max_ms=100, num_bins=50).get_stats(),
        },
        'sionna_latency': {
            'histogram': LatencyHistogram(min_ms=0, max_ms=100, num_bins=50).to_dict(),
            'stats': LatencyHistogram(min_ms=0, max_ms=100, num_bins=50).get_stats(),
        },
        'gan_latency': {
            'histogram': LatencyHistogram(min_ms=0, max_ms=50, num_bins=40).to_dict(),
            'stats': LatencyHistogram(min_ms=0, max_ms=50, num_bins=40).get_stats(),
        },
    }


def _empty_latency_breakdown() -> Dict[str, Any]:
    return {
        'decision_logic': {'mean_ms': 0.0, 'std_ms': 0.0, 'max_ms': 0.0},
        'gan_prediction': {'mean_ms': 0.0, 'std_ms': 0.0, 'max_ms': 0.0, 'samples': 0},
        'sionna_simulation': {'mean_ms': 0.0, 'std_ms': 0.0, 'max_ms': 0.0},
    }


def _latest_model_summary(events: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not events:
        return None
    latest = events[-1]
    model_name = latest.get('model_name') or latest.get('model_mode') or 'unknown'
    model_mode = latest.get('model_mode')
    if model_mode:
        return {'name': model_name, 'mode': model_mode}
    return {'name': model_name}


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/realtime-stats', methods=['GET'])
def get_realtime_stats():
    """Live performance metrics derived from raw events."""
    window = request.args.get('window', 50, type=int)
    try:
        events = _fetch_recent_events(window)
    except Exception as exc:
        logger.warning(f"[Dashboard] Failed to fetch raw events: {exc}")
        return jsonify({'error': f'Failed to fetch raw metrics: {exc}'}), 503

    stats = summarize_realtime_stats(events, window_size=window) if events else _empty_realtime_stats()
    current_model = _latest_model_summary(events)
    if current_model is not None:
        stats['current_model'] = current_model
    return jsonify(stats)


@app.route('/api/latency-histograms', methods=['GET'])
def get_latency_histograms():
    """Get latency histograms for all latency types from raw events."""
    window = request.args.get('window', 500, type=int)
    try:
        events = _fetch_recent_events(window)
    except Exception as exc:
        logger.warning(f"[Dashboard] Failed to fetch raw events: {exc}")
        return jsonify({'error': f'Failed to fetch raw metrics: {exc}'}), 503

    histograms = summarize_latency_histograms(events) if events else _empty_latency_histograms()
    return jsonify(histograms)


@app.route('/api/latency-breakdown', methods=['GET'])
def get_latency_breakdown():
    """Get latency breakdown analysis from raw events."""
    window = request.args.get('window', 500, type=int)
    try:
        events = _fetch_recent_events(window)
    except Exception as exc:
        logger.warning(f"[Dashboard] Failed to fetch raw events: {exc}")
        return jsonify({'error': f'Failed to fetch raw metrics: {exc}'}), 503

    breakdown = summarize_latency_breakdown(events) if events else _empty_latency_breakdown()
    return jsonify(breakdown)


@app.route('/api/flow/<int:flow_id>/stats', methods=['GET'])
def get_flow_stats(flow_id):
    """Per-flow metrics derived from raw events."""
    window = request.args.get('window', 5000, type=int)
    try:
        events = _fetch_recent_events(window)
    except Exception as exc:
        logger.warning(f"[Dashboard] Failed to fetch raw events: {exc}")
        return jsonify({'error': f'Failed to fetch raw metrics: {exc}'}), 503

    flow_events = [event for event in events if int(event.get('flow_id', -1)) == flow_id]
    if not flow_events:
        return jsonify({})

    return jsonify({
        'flow_id': flow_id,
        'decisions': len(flow_events),
        'avg_prediction_error': float(sum(float(event.get('prediction_error', 0.0)) for event in flow_events) / len(flow_events)),
        'avg_throughput': float(sum(float(event.get('throughput', 0.0)) for event in flow_events) / len(flow_events)),
        'avg_bler': float(sum(float(event.get('bler', 0.0)) for event in flow_events) / len(flow_events)),
        'avg_decision_latency': float(sum(float(event.get('decision_latency_ms', 0.0)) for event in flow_events) / len(flow_events)),
        'modulation_distribution': {
            mod: sum(1 for event in flow_events if event.get('chosen_modulation') == mod or event.get('selected_modulation') == mod)
            for mod in ['qam4', 'qam16', 'qam64', 'qam256']
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check for the dashboard and metrics server bridge."""
    try:
        response = requests.get(f"{_metrics_server_url()}/health", timeout=1)
        response.raise_for_status()
        metrics_health = response.json()
        return jsonify({
            'status': 'ok',
            'metrics_server_ready': True,
            'analyzer_ready': True,
            'event_count': metrics_health.get('event_count', 0),
        })
    except Exception as exc:
        logger.warning(f"[Dashboard] Metrics server health check failed: {exc}")
        return jsonify({
            'status': 'degraded',
            'metrics_server_ready': False,
            'analyzer_ready': False,
            'error': str(exc),
        })


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


def run_dashboard(analyzer=None, host='0.0.0.0', port=5000, debug=False, metrics_server_url: str = DEFAULT_METRICS_SERVER_URL):
    """Start the dashboard server"""
    if analyzer is not None:
        set_analyzer(analyzer)
    set_metrics_server_url(metrics_server_url)
    logger.info(f"[Dashboard] Using metrics server: {_metrics_server_url()}")
    logger.info(f"[Dashboard] Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_dashboard(debug=True)
