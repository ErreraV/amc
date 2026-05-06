#!/usr/bin/env python3
"""Metrics server: ingest raw events and expose dashboard-compatible APIs
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
import threading
import time
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In-memory event buffer
EVENT_BUFFER_MAX = 5000
_events = deque(maxlen=EVENT_BUFFER_MAX)
_events_lock = threading.Lock()

# Simple latency histogram helper
class LatencyHistogram:
    def __init__(self, min_ms=0, max_ms=200, num_bins=50):
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.num_bins = num_bins
        self.bin_edges = np.linspace(min_ms, max_ms, num_bins + 1)
        self.bins = np.zeros(num_bins, dtype=int)

    def add_samples(self, samples):
        if len(samples) == 0:
            return
        samples = np.clip(np.array(samples), self.min_ms, self.max_ms - 0.0001)
        idx = ((samples - self.min_ms) / (self.max_ms - self.min_ms) * self.num_bins).astype(int)
        for i in idx:
            if 0 <= i < self.num_bins:
                self.bins[i] += 1

    def to_dict(self):
        bin_centers = ((self.bin_edges[:-1] + self.bin_edges[1:]) / 2).tolist()
        return {
            'bin_centers': bin_centers,
            'bin_edges': self.bin_edges.tolist(),
            'counts': self.bins.tolist(),
            'num_bins': int(self.num_bins)
        }

# Append event
@app.route('/ingest_event', methods=['POST'])
def ingest_event():
    try:
        event = request.get_json(force=True)
        with _events_lock:
            _events.append(event)
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        logger.exception('Failed ingest event')
        return jsonify({'error': str(e)}), 500

# Dashboard-compatible endpoints
@app.route('/api/realtime-stats', methods=['GET'])
def get_realtime_stats():
    window = int(request.args.get('window', 50))
    with _events_lock:
        recent = list(_events)[-window:]
    if not recent:
        return jsonify({}), 200

    # Compute simple aggregates
    prediction_errors = [e.get('prediction_error', 0.0) for e in recent if 'prediction_error' in e]
    prediction_accuracy = 0.0
    if prediction_errors:
        prediction_accuracy = float(sum(1 for x in prediction_errors if x <= 3.0) / len(prediction_errors))

    avg_latency = float(np.mean([e.get('decision_latency_ms', 0.0) for e in recent]))
    avg_throughput = float(np.mean([e.get('throughput', 0.0) or 0.0 for e in recent]))
    avg_bler = float(np.mean([e.get('bler', 0.0) or 0.0 for e in recent]))
    avg_ber = float(np.mean([e.get('ber', 1e-6) or 1e-6 for e in recent]))
    success_rate = float(sum(1 for e in recent if e.get('success')) / len(recent))

    decision_method_distribution = {
        'gan_preselected': sum(1 for e in recent if e.get('decision_method', '').startswith('preselected')),
        'rl_fallback': sum(1 for e in recent if 'rl' in e.get('decision_method', ''))
    }

    latest_event = recent[-1]
    current_model = {
        'mode': latest_event.get('model_mode', 'unknown'),
        'name': latest_event.get('model_name', 'Unknown model')
    }

    resp = {
        'window_size': len(recent),
        'avg_prediction_error': float(np.mean(prediction_errors)) if prediction_errors else 0.0,
        'prediction_accuracy': prediction_accuracy,
        'avg_latency_ms': avg_latency,
        'avg_throughput': avg_throughput,
        'avg_bler': avg_bler,
        'avg_ber': avg_ber,
        'success_rate': success_rate,
        'last_sample_time': recent[-1].get('timestamp', time.time()),
        'decision_method_distribution': decision_method_distribution,
        'current_model': current_model
    }
    return jsonify(resp), 200

@app.route('/api/latency-histograms', methods=['GET'])
def get_latency_histograms():
    with _events_lock:
        snapshot = list(_events)
    if not snapshot:
        return jsonify({}), 200

    decision_lat = [e.get('decision_latency_ms', 0.0) for e in snapshot]
    sionna_lat = [e.get('sionna_latency_ms', 0.0) for e in snapshot]
    gan_lat = [e.get('gan_latency_ms', 0.0) for e in snapshot if e.get('gan_latency_ms', 0.0) > 0]

    d_hist = LatencyHistogram(min_ms=0, max_ms=200, num_bins=50)
    s_hist = LatencyHistogram(min_ms=0, max_ms=200, num_bins=50)
    g_hist = LatencyHistogram(min_ms=0, max_ms=100, num_bins=40)

    d_hist.add_samples(decision_lat)
    s_hist.add_samples(sionna_lat)
    g_hist.add_samples(gan_lat)

    resp = {
        'decision_latency': {'histogram': d_hist.to_dict(), 'stats': _hist_stats(decision_lat)},
        'sionna_latency': {'histogram': s_hist.to_dict(), 'stats': _hist_stats(sionna_lat)},
        'gan_latency': {'histogram': g_hist.to_dict(), 'stats': _hist_stats(gan_lat)}
    }
    return jsonify(resp), 200

@app.route('/api/latency-breakdown', methods=['GET'])
def get_latency_breakdown():
    with _events_lock:
        recent = list(_events)
    if not recent:
        return jsonify({}), 200

    decision_logic_times = [e.get('decision_latency_ms', 0.0) - e.get('sionna_latency_ms', 0.0) for e in recent]
    gan_times = [e.get('gan_latency_ms', 0.0) for e in recent if e.get('gan_latency_ms', 0.0) > 0]
    sionna_times = [e.get('sionna_latency_ms', 0.0) for e in recent]

    resp = {
        'decision_logic': {'mean_ms': float(np.mean(decision_logic_times)), 'std_ms': float(np.std(decision_logic_times)), 'max_ms': float(max(decision_logic_times))},
        'gan_prediction': {'mean_ms': float(np.mean(gan_times)) if gan_times else 0.0, 'std_ms': float(np.std(gan_times)) if gan_times else 0.0, 'max_ms': float(max(gan_times)) if gan_times else 0.0, 'samples': len(gan_times)},
        'sionna_simulation': {'mean_ms': float(np.mean(sionna_times)), 'std_ms': float(np.std(sionna_times)), 'max_ms': float(max(sionna_times))}
    }
    return jsonify(resp), 200

def _hist_stats(samples):
    if not samples:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'count': 0}
    arr = np.array(samples)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'count': int(len(arr))
    }

def run_metrics_server(host='127.0.0.1', port=5001):
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_metrics_server()
