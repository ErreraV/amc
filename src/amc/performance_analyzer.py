"""Real-time performance analysis with latency histograms.

Description:
    Utilities and data structures for collecting, summarizing and exposing
    real-time performance metrics (latency histograms, error summaries, etc.).

How to use:
    Import from other modules to record and summarize `PerformanceMetrics`:
        from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
"""
import time
import threading
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List
import json


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    timestamp: float = field(default_factory=time.time)
    flow_id: int = 0
    decision_method: str = ""
    predicted_snr: float = 0.0
    actual_snr: float = 0.0
    prediction_error: float = 0.0
    selected_modulation: str = ""
    ber: float = 0.0
    bler: float = 0.0
    throughput: float = 0.0
    decision_latency_ms: float = 0.0  # Total time to make decision
    sionna_latency_ms: float = 0.0    # Time for Sionna response
    gan_latency_ms: float = 0.0       # Time for GAN prediction
    success: bool = False


def _recent_records(records, window_size: int):
    recent = list(records)
    if window_size > 0:
        return recent[-window_size:]
    return recent


def _record_value(record: Any, field_name: str, default: Any = 0.0) -> Any:
    if isinstance(record, dict):
        value = record.get(field_name, default)
    else:
        value = getattr(record, field_name, default)
    return default if value is None else value


def _decision_method_matches(record: Any, needle: str) -> bool:
    return needle in str(_record_value(record, 'decision_method', ''))


def summarize_realtime_stats(records, window_size: int = 50) -> Dict:
    recent = _recent_records(records, window_size)

    if not recent:
        return {}

    prediction_errors = [float(_record_value(record, 'prediction_error', 0.0)) for record in recent]
    avg_ber_samples = [float(_record_value(record, 'ber', 1e-6)) for record in recent]
    avg_ber_samples = [sample if sample > 0 else 1e-6 for sample in avg_ber_samples]

    return {
        'window_size': len(recent),
        'avg_prediction_error': float(np.mean(prediction_errors)),
        'prediction_accuracy': float(sum(1 for error in prediction_errors if error <= 3.0) / len(recent)),
        'avg_latency_ms': float(np.mean([float(_record_value(record, 'decision_latency_ms', 0.0)) for record in recent])),
        'avg_throughput': float(np.mean([float(_record_value(record, 'throughput', 0.0)) for record in recent])),
        'avg_bler': float(np.mean([float(_record_value(record, 'bler', 0.0)) for record in recent])),
        'avg_ber': float(np.mean(avg_ber_samples)),
        'success_rate': float(sum(1 for record in recent if bool(_record_value(record, 'success', False))) / len(recent)),
        'last_sample_time': float(_record_value(recent[-1], 'timestamp', 0.0)),
        'decision_method_distribution': {
            'gan_preselected': sum(1 for record in recent if _decision_method_matches(record, 'preselected')),
            'rl_fallback': sum(1 for record in recent if _decision_method_matches(record, 'rl_agent') or _decision_method_matches(record, 'rl')),
        }
    }


def summarize_latency_breakdown(records) -> Dict:
    recent = list(records)

    if not recent:
        return {}

    decision_logic_times = [
        float(_record_value(record, 'decision_latency_ms', 0.0)) - float(_record_value(record, 'sionna_latency_ms', 0.0))
        for record in recent
    ]
    gan_times = [float(_record_value(record, 'gan_latency_ms', 0.0)) for record in recent if float(_record_value(record, 'gan_latency_ms', 0.0)) > 0]
    sionna_times = [float(_record_value(record, 'sionna_latency_ms', 0.0)) for record in recent]

    return {
        'decision_logic': {
            'mean_ms': float(np.mean(decision_logic_times)),
            'std_ms': float(np.std(decision_logic_times)),
            'max_ms': float(np.max(decision_logic_times)) if decision_logic_times else 0,
        },
        'gan_prediction': {
            'mean_ms': float(np.mean(gan_times)) if gan_times else 0,
            'std_ms': float(np.std(gan_times)) if gan_times else 0,
            'max_ms': float(np.max(gan_times)) if gan_times else 0,
            'samples': len(gan_times),
        },
        'sionna_simulation': {
            'mean_ms': float(np.mean(sionna_times)),
            'std_ms': float(np.std(sionna_times)),
            'max_ms': float(np.max(sionna_times)) if sionna_times else 0,
        }
    }


def summarize_latency_histograms(records) -> Dict:
    recent = list(records)

    if not recent:
        return {}

    decision_hist = LatencyHistogram(min_ms=0, max_ms=200, num_bins=50)
    sionna_hist = LatencyHistogram(min_ms=0, max_ms=200, num_bins=50)
    gan_hist = LatencyHistogram(min_ms=0, max_ms=100, num_bins=40)

    decision_samples = [float(_record_value(record, 'decision_latency_ms', 0.0)) for record in recent]
    sionna_samples = [float(_record_value(record, 'sionna_latency_ms', 0.0)) for record in recent]
    gan_samples = [float(_record_value(record, 'gan_latency_ms', 0.0)) for record in recent if float(_record_value(record, 'gan_latency_ms', 0.0)) > 0]

    decision_hist.add_samples(decision_samples)
    sionna_hist.add_samples(sionna_samples)
    gan_hist.add_samples(gan_samples)

    return {
        'decision_latency': {
            'histogram': decision_hist.to_dict(),
            'stats': decision_hist.get_stats(),
        },
        'sionna_latency': {
            'histogram': sionna_hist.to_dict(),
            'stats': sionna_hist.get_stats(),
        },
        'gan_latency': {
            'histogram': gan_hist.to_dict(),
            'stats': gan_hist.get_stats(),
        },
    }


def summarize_quality_breakdown(records, window_size: int = 50) -> Dict:
    recent = _recent_records(records, window_size)

    if not recent:
        return {}

    def _summarize(samples, prefix=''):
        if not samples:
            return {
                'count': 0,
                'avg_ber': 0.0,
                'avg_bler': 0.0,
                'avg_throughput': 0.0,
                'success_rate': 0.0,
            }

        return {
            'count': len(samples),
            'avg_ber': float(np.nanmean([_record_value(record, f'{prefix}ber', np.nan) for record in samples])),
            'avg_bler': float(np.nanmean([_record_value(record, f'{prefix}bler', np.nan) for record in samples])),
            'avg_throughput': float(np.nanmean([_record_value(record, f'{prefix}throughput', np.nan) for record in samples])),
            'success_rate': float(sum(1 for record in samples if bool(_record_value(record, 'success', False))) / len(samples)),
        }

    paired_samples = [record for record in recent if 'heuristic_bler' in record or 'actual_bler' in record]

    return {
        'window_size': len(recent),
        'overall': _summarize(recent),
        'by_source': {
            'sionna': _summarize(paired_samples, prefix='actual_'),
            'heuristic': _summarize(paired_samples, prefix='heuristic_'),
        },
        'source_counts': {
            'sionna': sum(1 for record in paired_samples if not np.isnan(_record_value(record, 'actual_bler', np.nan))),
            'heuristic': sum(1 for record in paired_samples if not np.isnan(_record_value(record, 'heuristic_bler', np.nan))),
        },
    }


class LatencyHistogram:
    """Tracks latency distribution with histogram bins"""
    
    def __init__(self, min_ms=0, max_ms=200, num_bins=50):
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.num_bins = num_bins
        self.bin_width = (max_ms - min_ms) / num_bins
        self.bins = np.zeros(num_bins)
        self.all_samples = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def add_sample(self, latency_ms: float):
        """Add a latency sample to histogram"""
        with self.lock:
            self.all_samples.append(latency_ms)
            
            # Clamp to bin range
            clamped = np.clip(latency_ms, self.min_ms, self.max_ms - 0.01)
            bin_idx = int((clamped - self.min_ms) / self.bin_width)
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            self.bins[bin_idx] += 1
    
    def add_samples(self, latency_ms_list: List[float]):
        """Add multiple latency samples"""
        for latency_ms in latency_ms_list:
            self.add_sample(latency_ms)
    
    def get_histogram(self) -> Dict:
        """Get histogram data for visualization"""
        with self.lock:
            bin_edges = np.linspace(self.min_ms, self.max_ms, self.num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            return {
                'bin_centers': bin_centers.tolist(),
                'bin_edges': bin_edges.tolist(),
                'counts': self.bins.tolist(),
                'bin_width': self.bin_width
            }

    def to_dict(self) -> Dict:
        histogram = self.get_histogram()
        histogram['num_bins'] = int(self.num_bins)
        return histogram
    
    def get_stats(self) -> Dict:
        """Get statistical summary"""
        with self.lock:
            samples = list(self.all_samples)
        
        if not samples:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        return {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'p50': float(np.percentile(samples, 50)),
            'p95': float(np.percentile(samples, 95)),
            'p99': float(np.percentile(samples, 99)),
            'count': len(samples)
        }


class RealtimeAnalyzer:
    """Comprehensive real-time performance analysis"""
    
    def __init__(self, window_size=200):
        self.metrics_buffer = deque(maxlen=window_size)
        
        # Latency histograms
        self.decision_latency_hist = LatencyHistogram(min_ms=0, max_ms=100, num_bins=50)
        self.sionna_latency_hist = LatencyHistogram(min_ms=0, max_ms=100, num_bins=50)
        self.gan_latency_hist = LatencyHistogram(min_ms=0, max_ms=50, num_bins=40)
        
        self.flow_stats = defaultdict(lambda: {
            'predictions_accurate': 0,
            'predictions_total': 0,
            'avg_bler': 0.0,
            'avg_throughput': 0.0,
            'decision_count': 0
        })
        self.lock = threading.Lock()
    
    def record_decision(self, metrics: PerformanceMetrics):
        """Record a decision with all metrics"""
        with self.lock:
            self.metrics_buffer.append(metrics)
        
        # Update latency histograms
        self.decision_latency_hist.add_sample(metrics.decision_latency_ms)
        self.sionna_latency_hist.add_sample(metrics.sionna_latency_ms)
        if metrics.gan_latency_ms > 0:
            self.gan_latency_hist.add_sample(metrics.gan_latency_ms)
    
    def get_real_time_stats(self, window_size=50) -> Dict:
        """Get last N decisions stats"""
        with self.lock:
            recent = list(self.metrics_buffer)

        return summarize_realtime_stats(recent, window_size=window_size)

    def get_latency_histograms(self) -> Dict:
        """Get all latency histograms"""
        with self.lock:
            recent = list(self.metrics_buffer)

        return summarize_latency_histograms(recent)

    def get_flow_stats(self, flow_id: int) -> Dict:
        """Per-flow performance"""
        with self.lock:
            flow_metrics = [m for m in self.metrics_buffer if m.flow_id == flow_id]

        if not flow_metrics:
            return {}

        return {
            'flow_id': flow_id,
            'decisions': len(flow_metrics),
            'avg_prediction_error': float(np.mean([m.prediction_error for m in flow_metrics])),
            'avg_throughput': float(np.mean([m.throughput for m in flow_metrics])),
            'avg_bler': float(np.mean([m.bler for m in flow_metrics])),
            'avg_decision_latency': float(np.mean([m.decision_latency_ms for m in flow_metrics])),
            'modulation_distribution': {
                mod: sum(1 for m in flow_metrics if m.selected_modulation == mod)
                for mod in ['qam4', 'qam16', 'qam64', 'qam256']
            }
        }

    def get_latency_breakdown(self) -> Dict:
        """Analyze where time is spent"""
        with self.lock:
            metrics = list(self.metrics_buffer)

        return summarize_latency_breakdown(metrics)


if __name__ == '__main__':
    # Test the analyzer
    analyzer = RealtimeAnalyzer()
    
    # Simulate some metrics
    for i in range(100):
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
            decision_latency_ms=np.abs(np.random.normal(30, 10)),
            sionna_latency_ms=np.abs(np.random.normal(25, 8)),
            gan_latency_ms=np.abs(np.random.normal(5, 2)) if i % 2 == 0 else 0,
            success=np.random.random() > 0.1
        )
        analyzer.record_decision(metrics)
    
    # Print results
    print("Real-time Stats:")
    print(json.dumps(analyzer.get_real_time_stats(), indent=2))
    print("\nLatency Histograms:")
    print(json.dumps(analyzer.get_latency_histograms(), indent=2))
    print("\nLatency Breakdown:")
    print(json.dumps(analyzer.get_latency_breakdown(), indent=2))
