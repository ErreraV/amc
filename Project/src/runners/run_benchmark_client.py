#!/usr/bin/env python3
"""Standalone benchmark client for AMC metrics server.

This client can run in parallel with the dashboard or independently.
It polls the metrics server APIs, stores benchmark snapshots, and saves plots.
"""

import argparse
import json
import logging
import signal
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Sample:
    elapsed_s: float
    wall_time: float
    avg_latency_ms: float
    avg_throughput: float
    success_rate: float
    avg_ber: float
    avg_bler: float
    prediction_accuracy: float
    avg_prediction_error: float
    window_size: int
    model_mode: str
    model_name: str


class BenchmarkClient:
    def __init__(
        self,
        metrics_base_url: str,
        poll_interval_s: float,
        window_size: int,
        output_dir: Path,
        duration_s: Optional[float] = None,
    ):
        self.metrics_base_url = metrics_base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.window_size = window_size
        self.output_dir = output_dir
        self.duration_s = duration_s
        self.samples: List[Sample] = []
        self.latest_histograms: Dict = {}
        self.latest_breakdown: Dict = {}
        self._stop = False

    def stop(self, *_args):
        self._stop = True

    def _get_json(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.metrics_base_url}{path}"
        response = requests.get(url, params=params, timeout=2.0)
        response.raise_for_status()
        return response.json()

    def _poll_once(self, start_time: float):
        stats = self._get_json('/api/realtime-stats', params={'window': self.window_size})
        if not stats:
            return

        self.latest_histograms = self._get_json('/api/latency-histograms')
        self.latest_breakdown = self._get_json('/api/latency-breakdown')

        model = stats.get('current_model', {})
        sample = Sample(
            elapsed_s=time.time() - start_time,
            wall_time=time.time(),
            avg_latency_ms=float(stats.get('avg_latency_ms', 0.0)),
            avg_throughput=float(stats.get('avg_throughput', 0.0)),
            success_rate=float(stats.get('success_rate', 0.0)),
            avg_ber=float(stats.get('avg_ber', 0.0)),
            avg_bler=float(stats.get('avg_bler', 0.0)),
            prediction_accuracy=float(stats.get('prediction_accuracy', 0.0)),
            avg_prediction_error=float(stats.get('avg_prediction_error', 0.0)),
            window_size=int(stats.get('window_size', 0)),
            model_mode=str(model.get('mode', 'unknown')),
            model_name=str(model.get('name', 'Unknown model')),
        )
        self.samples.append(sample)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        logger.info("Benchmark client started")
        logger.info("Metrics server: %s", self.metrics_base_url)
        logger.info("Poll interval: %.2fs | Window: %d", self.poll_interval_s, self.window_size)
        if self.duration_s:
            logger.info("Duration: %.1fs", self.duration_s)
        else:
            logger.info("Duration: until interrupted (Ctrl+C)")

        while not self._stop:
            if self.duration_s is not None and (time.time() - start_time) >= self.duration_s:
                logger.info("Duration reached, stopping benchmark collection")
                break

            try:
                self._poll_once(start_time)
                if self.samples:
                    latest = self.samples[-1]
                    logger.info(
                        "t=%.1fs latency=%.2fms throughput=%.2fMbps success=%.1f%% model=%s",
                        latest.elapsed_s,
                        latest.avg_latency_ms,
                        latest.avg_throughput,
                        latest.success_rate * 100.0,
                        latest.model_name,
                    )
            except requests.RequestException as exc:
                logger.warning("Metrics server request failed: %s", exc)
            except Exception as exc:
                logger.warning("Unexpected polling error: %s", exc)

            time.sleep(self.poll_interval_s)

        self._save_outputs()

    def _save_outputs(self):
        if not self.samples:
            logger.warning("No samples collected; nothing to save")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # include final model information in folder name for easier identification
        final_model = self.samples[-1].model_name if self.samples else 'unknown'
        # sanitize model name for filesystem
        safe_model = ''.join(c if (c.isalnum() or c in ('_', '-')) else '_' for c in final_model).strip('_')[:60]
        run_dir = self.output_dir / f'benchmark_{timestamp}_{safe_model}'
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'metadata': {
                'metrics_base_url': self.metrics_base_url,
                'poll_interval_s': self.poll_interval_s,
                'window_size': self.window_size,
                'sample_count': len(self.samples),
                'start_wall_time': self.samples[0].wall_time,
                'end_wall_time': self.samples[-1].wall_time,
                'elapsed_s': self.samples[-1].elapsed_s,
                'final_model_mode': self.samples[-1].model_mode,
                'final_model_name': self.samples[-1].model_name,
            },
            'time_series': [asdict(s) for s in self.samples],
            'latency_histograms': self.latest_histograms,
            'latency_breakdown': self.latest_breakdown,
        }

        summary_file = run_dir / 'benchmark_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        self._save_timeseries_plot(run_dir / 'timeseries.png')
        self._save_quality_plot(run_dir / 'quality.png')

        logger.info("Saved benchmark artifacts to %s", run_dir)
        logger.info("- %s", summary_file.name)
        logger.info("- timeseries.png")
        logger.info("- quality.png")

    def _save_timeseries_plot(self, output_path: Path):
        t = [s.elapsed_s for s in self.samples]
        latency = [s.avg_latency_ms for s in self.samples]
        throughput = [s.avg_throughput for s in self.samples]
        success = [s.success_rate * 100.0 for s in self.samples]

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        fig.suptitle(f"AMC Benchmark Time Series - {self.samples[-1].model_name}")

        axes[0].plot(t, latency, color='#1f77b4', linewidth=1.8)
        axes[0].set_ylabel('Latency (ms)')
        axes[0].grid(alpha=0.3)

        axes[1].plot(t, throughput, color='#2ca02c', linewidth=1.8)
        axes[1].set_ylabel('Throughput (Mbps)')
        axes[1].grid(alpha=0.3)

        axes[2].plot(t, success, color='#ff7f0e', linewidth=1.8)
        axes[2].set_ylabel('Success Rate (%)')
        axes[2].set_xlabel('Elapsed Time (s)')
        axes[2].set_ylim(0, 100)
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _save_quality_plot(self, output_path: Path):
        t = [s.elapsed_s for s in self.samples]
        ber = [s.avg_ber for s in self.samples]
        bler = [s.avg_bler for s in self.samples]
        pred_acc = [s.prediction_accuracy * 100.0 for s in self.samples]
        pred_err = [s.avg_prediction_error for s in self.samples]

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        fig.suptitle('AMC Link Quality and Prediction Performance')

        axes[0].plot(t, ber, label='Avg BER', color='#d62728', linewidth=1.5)
        axes[0].plot(t, bler, label='Avg BLER', color='#9467bd', linewidth=1.5)
        axes[0].set_ylabel('Rate')
        axes[0].set_yscale('log')
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(t, pred_acc, label='Prediction Accuracy (%)', color='#17becf', linewidth=1.5)
        axes[1].plot(t, pred_err, label='Avg Prediction Error (dB)', color='#8c564b', linewidth=1.5)
        axes[1].set_ylabel('Prediction Metrics')
        axes[1].set_xlabel('Elapsed Time (s)')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Standalone AMC benchmark plotting client')
    parser.add_argument('--metrics-url', default='http://127.0.0.1:5001', help='Metrics server base URL')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds')
    parser.add_argument('--window', type=int, default=50, help='Realtime stats window size')
    parser.add_argument('--duration', type=float, default=None, help='Optional collection duration in seconds')
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).parent.parent.parent / 'benchmarks' / 'results'),
        help='Directory where benchmark artifacts are written',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = BenchmarkClient(
        metrics_base_url=args.metrics_url,
        poll_interval_s=args.interval,
        window_size=args.window,
        output_dir=Path(args.output_dir),
        duration_s=args.duration,
    )

    signal.signal(signal.SIGINT, client.stop)
    signal.signal(signal.SIGTERM, client.stop)

    client.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
