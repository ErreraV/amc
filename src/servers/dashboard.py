#!/usr/bin/env python3
"""
Flask dashboard for real-time latency visualization
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from ..amc.performance_analyzer import (
    RealtimeAnalyzer,
    summarize_latency_breakdown,
    summarize_latency_histograms,
    summarize_realtime_stats,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Store analyzer in Flask config (thread-safe and recommended)
def set_analyzer(analyzer: RealtimeAnalyzer):
    """Set the analyzer instance in Flask config"""
    app.config['ANALYZER'] = analyzer
    logger.info(f"[Dashboard] Analyzer set in config: {analyzer}")


def set_metrics_server_url(metrics_server_url: str):
    """Set the metrics server URL for process-isolated dashboard mode."""
    app.config['METRICS_SERVER_URL'] = metrics_server_url.rstrip('/')
    logger.info(f"[Dashboard] Metrics server URL set in config: {app.config['METRICS_SERVER_URL']}")


def _record_value(record: Any, field_name: str, default: Any = 0.0) -> Any:
    if isinstance(record, dict):
        value = record.get(field_name, default)
    else:
        value = getattr(record, field_name, default)
    return default if value is None else value


def _latest_event(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return events[-1] if events else None


def _fetch_events_from_metrics_server(window: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    metrics_server_url = app.config.get('METRICS_SERVER_URL')
    if not metrics_server_url:
        return None

    try:
        params = {'window': window} if window is not None else None
        response = requests.get(f"{metrics_server_url}/api/events", params=params, timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return payload
        return []
    except requests.RequestException as exc:
        logger.warning(f"[Dashboard] Failed to fetch metrics events: {exc}")
        return None


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


def _current_model_from_event(event: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not event:
        return {'mode': 'unknown', 'name': 'Unknown model'}
    return {
        'mode': str(_record_value(event, 'model_mode', 'unknown')),
        'name': str(_record_value(event, 'model_name', 'Unknown model')),
    }


def _flow_stats_from_events(events: List[Dict[str, Any]], flow_id: int) -> Dict[str, Any]:
    flow_events = [event for event in events if int(_record_value(event, 'flow_id', -1)) == flow_id]
    if not flow_events:
        return {}

    def _mean(field_name: str) -> float:
        values = [float(_record_value(event, field_name, 0.0) or 0.0) for event in flow_events]
        return float(sum(values) / len(values)) if values else 0.0

    return {
        'flow_id': flow_id,
        'decisions': len(flow_events),
        'avg_prediction_error': _mean('prediction_error'),
        'avg_throughput': _mean('throughput'),
        'avg_bler': _mean('bler'),
        'avg_decision_latency': _mean('decision_latency_ms'),
        'modulation_distribution': {
            mod: sum(1 for event in flow_events if _record_value(event, 'chosen_modulation', '') == mod)
            for mod in ['qam4', 'qam16', 'qam64', 'qam256']
        },
    }


@app.before_request
def before_request():
    """Validate dashboard data source configuration."""
    analyzer = app.config.get('ANALYZER')
    metrics_server_url = app.config.get('METRICS_SERVER_URL')
    if analyzer is None and not metrics_server_url:
        logger.warning("[Dashboard] No analyzer or metrics server URL configured")
    return None


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/realtime-stats', methods=['GET'])
def get_realtime_stats():
    """Live performance metrics"""
    window = request.args.get('window', 50, type=int)
    analyzer = app.config.get('ANALYZER')
    latest_event: Optional[Dict[str, Any]] = None

    if analyzer is not None:
        stats = analyzer.get_real_time_stats(window)
        with analyzer.lock:
            latest_event = analyzer.metrics_buffer[-1] if analyzer.metrics_buffer else None
    else:
        events = _fetch_events_from_metrics_server(window)
        if events is None:
            return jsonify({'error': 'Analyzer not initialized and metrics server URL not configured'}), 503
        stats = summarize_realtime_stats(events, window_size=window)
        latest_event = _latest_event(events)

    stats = {**_empty_realtime_stats(), **dict(stats)}
    stats['current_model'] = _current_model_from_event(latest_event)
    return jsonify(stats)


@app.route('/api/latency-histograms', methods=['GET'])
def get_latency_histograms():
    """Get latency histograms for all latency types"""
    analyzer = app.config.get('ANALYZER')
    if analyzer is not None:
        histograms = analyzer.get_latency_histograms()
    else:
        events = _fetch_events_from_metrics_server()
        if events is None:
            return jsonify({'error': 'Analyzer not initialized and metrics server URL not configured'}), 503
        histograms = summarize_latency_histograms(events)
    return jsonify(histograms)


@app.route('/api/latency-breakdown', methods=['GET'])
def get_latency_breakdown():
    """Get latency breakdown analysis"""
    analyzer = app.config.get('ANALYZER')
    if analyzer is not None:
        breakdown = analyzer.get_latency_breakdown()
    else:
        events = _fetch_events_from_metrics_server()
        if events is None:
            return jsonify({'error': 'Analyzer not initialized and metrics server URL not configured'}), 503
        breakdown = summarize_latency_breakdown(events)
    return jsonify(breakdown)


@app.route('/api/flow/<int:flow_id>/stats', methods=['GET'])
def get_flow_stats(flow_id):
    """Per-flow metrics"""
    analyzer = app.config.get('ANALYZER')
    if analyzer is not None:
        stats = analyzer.get_flow_stats(flow_id)
    else:
        events = _fetch_events_from_metrics_server()
        if events is None:
            return jsonify({'error': 'Analyzer not initialized and metrics server URL not configured'}), 503
        stats = _flow_stats_from_events(events, flow_id)
    return jsonify(stats)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    analyzer = app.config.get('ANALYZER')
    return jsonify({'status': 'ok', 'analyzer_ready': analyzer is not None})


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


def run_dashboard(
    analyzer: Optional[RealtimeAnalyzer] = None,
    host='0.0.0.0',
    port=5000,
    debug=False,
    metrics_server_url: Optional[str] = None,
):
    """Start the dashboard server"""
    logger.info(f"[Dashboard] Initializing with analyzer: {analyzer}")

    if analyzer is not None:
        set_analyzer(analyzer)
        logger.info(f"[Dashboard] Analyzer stored in Flask config: {app.config.get('ANALYZER')}")

    if metrics_server_url is not None:
        set_metrics_server_url(metrics_server_url)

    logger.info(
        f"[Dashboard] Starting in {'metrics-server' if analyzer is None else 'analyzer'} mode on {host}:{port}"
    )
    logger.info(f"[Dashboard] Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    try:
        from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
    except ImportError:
        import sys

        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics  # type: ignore
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
