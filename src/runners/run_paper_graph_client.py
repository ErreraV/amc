#!/usr/bin/env python3
"""Paper-style benchmark client for AMC figure generation.

This client polls raw AMC events from the metrics server, bins them locally by
SNR, and produces plots matching the article's system-level evaluation:
throughput vs SNR, spectral efficiency vs SNR, and BLER vs SNR.
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RawEvent:
    timestamp: float
    snr: float
    modulation: str
    throughput_mbps: float
    bler: float
    ber: float
    success: bool
    source: str
    decision_method: str


@dataclass
class SnrBinSummary:
    bin_left: float
    bin_right: float
    center: float
    sample_count: int
    avg_throughput_mbps: float
    avg_spectral_efficiency_bps_per_hz: float
    avg_bler: float
    avg_ber: float
    success_rate: float


class PaperGraphClient:
    def __init__(
        self,
        metrics_base_url: str,
        window_size: int,
        output_dir: Path,
        bandwidth_mhz: float = 10.0,
        snr_bin_width_db: float = 2.0,
    ):
        self.metrics_base_url = metrics_base_url.rstrip("/")
        self.window_size = window_size
        self.output_dir = output_dir
        self.bandwidth_mhz = bandwidth_mhz
        self.snr_bin_width_db = snr_bin_width_db
        self.events: List[RawEvent] = []

    def _get_json(self, path: str, params: Optional[Dict] = None):
        url = f"{self.metrics_base_url}{path}"
        response = requests.get(url, params=params, timeout=3.0)
        response.raise_for_status()
        return response.json()

    def fetch_events(self):
        raw_events = self._get_json('/api/events', params={'window': self.window_size})
        self.events = []
        for item in raw_events:
            snr = item.get('snr', item.get('snr_db', None))
            if snr is None:
                continue
            throughput = item.get('throughput', item.get('throughput_effective', item.get('effective_throughput', 0.0)))
            self.events.append(
                RawEvent(
                    timestamp=float(item.get('timestamp', 0.0)),
                    snr=float(snr),
                    modulation=str(item.get('chosen_modulation', item.get('modulation', 'unknown'))),
                    throughput_mbps=float(throughput or 0.0),
                    bler=float(item.get('bler', item.get('actual_bler', 0.0)) or 0.0),
                    ber=float(item.get('ber', item.get('actual_ber', 0.0)) or 0.0),
                    success=bool(item.get('success', False)),
                    source=str(item.get('transmission_source', item.get('source', 'unknown'))),
                    decision_method=str(item.get('decision_method', 'unknown')),
                )
            )
        logger.info("Fetched %d raw events from %s/api/events", len(self.events), self.metrics_base_url)

    def _snr_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        snr_values = np.array([event.snr for event in self.events], dtype=float)
        if snr_values.size == 0:
            return np.array([]), np.array([])
        snr_min = np.floor(np.nanmin(snr_values) / self.snr_bin_width_db) * self.snr_bin_width_db
        snr_max = np.ceil(np.nanmax(snr_values) / self.snr_bin_width_db) * self.snr_bin_width_db
        if snr_min == snr_max:
            snr_max = snr_min + self.snr_bin_width_db
        edges = np.arange(snr_min, snr_max + self.snr_bin_width_db, self.snr_bin_width_db)
        if edges.size < 2:
            edges = np.array([snr_min, snr_max])
        centers = (edges[:-1] + edges[1:]) / 2.0
        return edges, centers

    def summarize_by_snr(self) -> List[SnrBinSummary]:
        if not self.events:
            return []

        edges, centers = self._snr_bins()
        if edges.size == 0:
            return []

        summaries: List[SnrBinSummary] = []
        for index in range(len(edges) - 1):
            left = float(edges[index])
            right = float(edges[index + 1])
            if index == len(edges) - 2:
                bin_events = [event for event in self.events if left <= event.snr <= right]
            else:
                bin_events = [event for event in self.events if left <= event.snr < right]

            if not bin_events:
                continue

            throughput = np.array([event.throughput_mbps for event in bin_events], dtype=float)
            bler = np.array([event.bler for event in bin_events], dtype=float)
            ber = np.array([event.ber for event in bin_events], dtype=float)
            success = np.array([1.0 if event.success else 0.0 for event in bin_events], dtype=float)
            spectral_efficiency = throughput / self.bandwidth_mhz

            summaries.append(
                SnrBinSummary(
                    bin_left=left,
                    bin_right=right,
                    center=float(centers[index]),
                    sample_count=len(bin_events),
                    avg_throughput_mbps=float(np.mean(throughput)),
                    avg_spectral_efficiency_bps_per_hz=float(np.mean(spectral_efficiency)),
                    avg_bler=float(np.mean(bler)),
                    avg_ber=float(np.mean(ber)),
                    success_rate=float(np.mean(success)),
                )
            )

        return summaries

    def _split_by_modulation(self) -> Dict[str, List[RawEvent]]:
        grouped: Dict[str, List[RawEvent]] = {}
        for event in self.events:
            grouped.setdefault(event.modulation, []).append(event)
        return grouped

    def _save_summary(self, run_dir: Path, summaries: Sequence[SnrBinSummary]):
        payload = {
            'metadata': {
                'metrics_base_url': self.metrics_base_url,
                'window_size': self.window_size,
                'bandwidth_mhz': self.bandwidth_mhz,
                'snr_bin_width_db': self.snr_bin_width_db,
                'sample_count': len(self.events),
                'start_wall_time': min((event.timestamp for event in self.events), default=0.0),
                'end_wall_time': max((event.timestamp for event in self.events), default=0.0),
            },
            'raw_events': [asdict(event) for event in self.events],
            'snr_binned_summary': [asdict(summary) for summary in summaries],
        }
        summary_file = run_dir / 'paper_graph_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return summary_file

    def _plot_throughput_vs_snr(self, run_dir: Path, summaries: Sequence[SnrBinSummary]):
        if not self.events:
            return
        x = [event.snr for event in self.events]
        y = [event.throughput_mbps for event in self.events]
        fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.6))
        axis.scatter(x, y, color='#1f77b4', s=20, alpha=0.6, label='Individual samples')
        axis.set_xlabel('SNR (dB)')
        axis.set_ylabel('Throughput (Mbps)')
        axis.set_title('Throughput across the SNR range')
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(run_dir / 'throughput_vs_snr.png', dpi=160)
        plt.close(fig)

    def _plot_spectral_efficiency(self, run_dir: Path, summaries: Sequence[SnrBinSummary]):
        if not self.events:
            return
        x = [event.snr for event in self.events]
        y = [event.throughput_mbps / self.bandwidth_mhz for event in self.events]
        fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.6))
        axis.scatter(x, y, color='#2ca02c', s=20, alpha=0.6, label='Individual samples')
        axis.set_xlabel('SNR (dB)')
        axis.set_ylabel('Spectral Efficiency (bps/Hz)')
        axis.set_title('Spectral efficiency across the SNR range')
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(run_dir / 'spectral_efficiency_vs_snr.png', dpi=160)
        plt.close(fig)

    def _plot_bler_vs_snr(self, run_dir: Path, summaries: Sequence[SnrBinSummary]):
        if not summaries:
            return
        bler_values = []
        snr_labels = []
        for i, summary in enumerate(summaries):
            # Collect BLER values for each SNR bin from raw events that fall in this bin
            if i == len(summaries) - 1:
                # Last bin uses <= for both sides
                bin_events = [e for e in self.events if summary.bin_left <= e.snr <= summary.bin_right]
            else:
                bin_events = [e for e in self.events if summary.bin_left <= e.snr < summary.bin_right]
            if bin_events:
                bler_values.append([event.bler for event in bin_events])
                snr_labels.append(f"{summary.center:.1f}")
        
        if not bler_values:
            return
        
        fig, axis = plt.subplots(1, 1, figsize=(8.2, 5.6))
        axis.boxplot(bler_values, labels=snr_labels, showmeans=True)
        axis.set_xlabel('SNR (dB)')
        axis.set_ylabel('BLER')
        axis.set_title('BLER versus SNR')
        axis.set_yscale('log')
        axis.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(run_dir / 'bler_vs_snr.png', dpi=160)
        plt.close(fig)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_events()
        summaries = self.summarize_by_snr()
        if not self.events:
            raise RuntimeError('No raw AMC events available from the metrics server')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.output_dir / f'paper_graphs_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        self._save_summary(run_dir, summaries)
        self._plot_throughput_vs_snr(run_dir, summaries)
        self._plot_spectral_efficiency(run_dir, summaries)
        self._plot_bler_vs_snr(run_dir, summaries)

        logger.info('Saved paper-style plots from raw events to %s', run_dir)
        logger.info('- throughput_vs_snr.png')
        logger.info('- spectral_efficiency_vs_snr.png')
        logger.info('- bler_vs_snr.png')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Paper-style AMC graph generator')
    parser.add_argument('--metrics-url', default='http://127.0.0.1:5001', help='Metrics server base URL')
    parser.add_argument('--window', type=int, default=5000, help='How many raw events to fetch')
    parser.add_argument('--bandwidth-mhz', type=float, default=10.0, help='Bandwidth used for spectral efficiency conversion')
    parser.add_argument('--snr-bin-width', type=float, default=2.0, help='SNR bin width in dB')
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).parent.parent.parent / 'benchmarks' / 'paper_graphs'),
        help='Directory where paper-style artifacts are written',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = PaperGraphClient(
        metrics_base_url=args.metrics_url,
        window_size=args.window,
        output_dir=Path(args.output_dir),
        bandwidth_mhz=args.bandwidth_mhz,
        snr_bin_width_db=args.snr_bin_width,
    )
    client.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())