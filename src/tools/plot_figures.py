#!/usr/bin/env python3
"""Fetch raw events from the metrics server and generate figures 8, 9, 10.

Usage examples:
    python -m tools.plot_figures --metrics-url http://127.0.0.1:5001 --outdir /tmp/plots
    python -m tools.plot_figures --metrics-url http://metrics:5001 --fig 8 9

Figures produced:
 - Fig 8: Decision latency histogram (decision_latency_ms)
 - Fig 9: Rolling prediction accuracy (prediction_error <= 3 dB)
 - Fig10: Throughput distribution per modulation (boxplot)
"""
from __future__ import annotations

import argparse
import os
import math
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def _first_present(event: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = event.get(key)
        if value is not None:
            return value
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _modulation_key(event: Dict[str, Any]) -> Optional[str]:
    mod = _first_present(event, 'selected_modulation', 'chosen_modulation', 'modulation')
    if mod is None:
        return None
    return str(mod).lower()


def _snr_key(event: Dict[str, Any]) -> Optional[float]:
    value = _first_present(event, 'actual_snr', 'current_snr', 'snr', 'predicted_snr')
    if value is None:
        return None
    return _to_float(value, default=None)  # type: ignore[arg-type]


def _throughput_key(event: Dict[str, Any]) -> Optional[float]:
    value = _first_present(event, 'throughput', 'actual_throughput', 'heuristic_throughput', 'throughput_effective')
    if value is None:
        return None
    return _to_float(value, default=None)  # type: ignore[arg-type]


def _bler_key(event: Dict[str, Any]) -> Optional[float]:
    # Prefer `ber` when present (user requested BER); fall back to BLER keys for compatibility
    value = _first_present(event, 'ber', 'bler', 'actual_bler', 'heuristic_bler')
    if value is None:
        return None
    return _to_float(value, default=None)  # type: ignore[arg-type]


def _bler_only_key(event: Dict[str, Any]) -> Optional[float]:
    # Prefer BLER fields first (for comparison plot), fall back to BER if BLER absent
    value = _first_present(event, 'bler', 'actual_bler', 'heuristic_bler', 'ber')
    if value is None:
        return None
    return _to_float(value, default=None)  # type: ignore[arg-type]


def _safe_name(value: Any, fallback: str = 'unknown') -> str:
    text = str(value).strip() if value is not None else fallback
    safe = ''.join(ch if (ch.isalnum() or ch in ('_', '-')) else '_' for ch in text).strip('_')
    return safe[:60] if safe else fallback


def _latest_model_name(events: List[Dict[str, Any]]) -> str:
    if not events:
        return 'unknown'
    latest = events[-1]
    return _safe_name(
        _first_present(latest, 'model_name', 'model_mode', 'decision_method', default='unknown'),
        fallback='unknown',
    )


def fetch_events(metrics_url: str, window: Optional[int] = None) -> List[Dict[str, Any]]:
    url = metrics_url.rstrip('/') + '/api/events'
    params = {}
    if window is not None:
        params['window'] = int(window)
    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    events = r.json()
    if not isinstance(events, list):
        raise RuntimeError('Metrics server did not return a list of events')
    return events


def fig8_decision_latency_hist(events: List[Dict[str, Any]], outpath: str):
    samples = [float(e.get('decision_latency_ms', 0.0)) for e in events if 'decision_latency_ms' in e]
    if not samples:
        raise RuntimeError('No decision_latency_ms samples found')

    plt.figure(figsize=(7, 4))
    plt.hist(samples, bins=50, color='#4c72b0', edgecolor='k', alpha=0.9)
    plt.xlabel('Decision latency (ms)')
    plt.ylabel('Count')
    plt.title('Figure 8 — Decision latency distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig11_bler_vs_snr_box(events: List[Dict[str, Any]], outpath: str):
    """BLER vs SNR binned boxplot (prefers BLER keys)"""
    grouped = defaultdict(list)
    for e in events:
        snr = _snr_key(e)
        bler = _bler_only_key(e)
        if snr is None or bler is None:
            continue
        if snr < 10.0:
            grouped['very_low'].append(max(bler, 1e-6))
        elif snr <= 15.0:
            grouped['low'].append(max(bler, 1e-6))
        elif snr <= 20.0:
            grouped['medium'].append(max(bler, 1e-6))
        elif snr <= 25.0:
            grouped['high'].append(max(bler, 1e-6))
        else:
            grouped['very_high'].append(max(bler, 1e-6))

    labels = ['very_low', 'low', 'medium', 'high', 'very_high']
    data = [grouped.get(name, []) for name in labels]

    plt.figure(figsize=(7, 4))
    b = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch in b['boxes']:
        patch.set_facecolor('#d3d3d3')

    plt.yscale('log')
    plt.ylabel('BLER (log scale)')
    plt.xlabel('SNR category')
    plt.title('Figure 11 — BLER versus SNR (binned)')
    try:
        plt.ylim(1e-3, 1e-1)
    except Exception:
        pass
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig9_prediction_accuracy(events: List[Dict[str, Any]], outpath: str, window: int = 50):
    # sort by timestamp
    events_sorted = sorted([e for e in events if 'prediction_error' in e and 'timestamp' in e], key=lambda x: x['timestamp'])
    if not events_sorted:
        raise RuntimeError('No events with prediction_error+timestamp found')

    errors = [float(e.get('prediction_error', 1e9)) for e in events_sorted]
    times = [float(e['timestamp']) for e in events_sorted]

    # rolling accuracy: fraction of errors <= 3.0 in sliding window
    thresh = 3.0
    acc = []
    avg_err = []
    for i in range(len(errors)):
        start = max(0, i - window + 1)
        window_errs = errors[start:i+1]
        acc.append(float(sum(1 for er in window_errs if er <= thresh)) / len(window_errs))
        avg_err.append(float(np.mean(window_errs)))

    # convert times to relative seconds
    t0 = times[0]
    rel_t = [t - t0 for t in times]

    plt.figure(figsize=(8, 4))
    plt.plot(rel_t, acc, label=f'Rolling accuracy (window={window})', color='#2ca02c')
    plt.ylabel('Prediction accuracy (fraction <= 3 dB)')
    plt.xlabel('Time (s)')
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)

    ax2 = plt.twinx()
    ax2.plot(rel_t, avg_err, label='Avg prediction error (dB)', color='#ff7f0e', alpha=0.8)
    ax2.set_ylabel('Average prediction error (dB)')

    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Figure 9 — Prediction accuracy over time')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig10_throughput_by_modulation(events: List[Dict[str, Any]], outpath: str):
    groups = defaultdict(list)
    for e in events:
        mod = e.get('chosen_modulation') or e.get('selected_modulation') or e.get('modulation')
        if mod is None:
            continue
        try:
            thr = float(e.get('throughput', 0.0))
        except Exception:
            thr = 0.0
        groups[mod].append(thr)

    if not groups:
        raise RuntimeError('No modulation/throughput pairs found')

    mods = sorted(groups.keys())
    data = [groups[m] for m in mods]

    plt.figure(figsize=(8, 4))
    b = plt.boxplot(data, labels=mods, patch_artist=True)
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    for patch, color in zip(b['boxes'], (colors * 10)):
        patch.set_facecolor(color)

    plt.ylabel('Throughput (Mbps)')
    plt.xlabel('Modulation')
    plt.title('Figure 10 — Throughput distribution per modulation')
    plt.grid(axis='y', alpha=0.3)
    # overlay means
    means = [np.mean(d) if d else 0.0 for d in data]
    x = range(1, len(means) + 1)
    plt.plot(x, means, marker='o', color='k', linestyle='--', label='mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig8_throughput_vs_snr(events: List[Dict[str, Any]], outpath: str):
    samples = [
        e for e in events
        if _snr_key(e) is not None and _throughput_key(e) is not None
    ]
    if not samples:
        raise RuntimeError('No SNR/throughput samples found')

    mod_colors = {
        'qam4': '#1f77b4',
        'qam16': '#ff7f0e',
        'qam64': '#2ca02c',
        'qam256': '#d62728'
    }

    plt.figure(figsize=(6, 4))
    for mod, color in mod_colors.items():
        xs = [_snr_key(e) for e in samples if _modulation_key(e) == mod]
        ys = [_throughput_key(e) for e in samples if _modulation_key(e) == mod]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if xs and ys:
            plt.scatter(xs, ys, s=18, alpha=0.8, label=mod, color=color, edgecolors='none')

    xs = [_snr_key(e) for e in samples if _modulation_key(e) is None]
    ys = [_throughput_key(e) for e in samples if _modulation_key(e) is None]
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if xs and ys:
        plt.scatter(xs, ys, s=12, alpha=0.5, color='#7f7f7f', label='unknown')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Figure 8 — Throughput across SNR (by modulation)')
    plt.legend(title='Modulation')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig9_spectral_efficiency(events: List[Dict[str, Any]], outpath: str):
    bits_map = {'qam4': 2.0, 'qam16': 4.0, 'qam64': 6.0, 'qam256': 8.0}
    samples = [
        e for e in events
        if _snr_key(e) is not None and _throughput_key(e) is not None
    ]
    if not samples:
        raise RuntimeError('No SNR/throughput samples found')

    plt.figure(figsize=(6, 3.2))
    for mod, bits in bits_map.items():
        xs = [_snr_key(e) for e in samples if _modulation_key(e) == mod]
        thr = [_throughput_key(e) for e in samples if _modulation_key(e) == mod]
        xs = [x for x in xs if x is not None]
        thr = [t for t in thr if t is not None]
        if xs and thr:
            spec_eff = [t / bits if bits > 0 else 0.0 for t in thr]
            plt.scatter(xs, spec_eff, s=18, alpha=0.8, label=mod)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectral efficiency (Mbps per bit)')
    plt.title('Figure 9 — Spectral efficiency across modulation schemes')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def fig10_bler_vs_snr_box(events: List[Dict[str, Any]], outpath: str):
    grouped = defaultdict(list)
    for e in events:
        snr = _snr_key(e)
        ber = _bler_key(e)
        if snr is None or ber is None:
            continue
        # User-specified SNR categories:
        # very_low: snr < 10.0 dB
        # low: 10.0 <= snr <= 15.0 dB
        # medium: 15.0 < = snr <= 20.0 dB
        # high: 20.0 < = snr <= 25.0 dB
        # very_high: snr > 25.0 dB
        if snr < 10.0:
            grouped['very_low'].append(max(ber, 1e-6))
        elif snr <= 15.0:
            grouped['low'].append(max(ber, 1e-6))
        elif snr <= 20.0:
            grouped['medium'].append(max(ber, 1e-6))
        elif snr <= 25.0:
            grouped['high'].append(max(ber, 1e-6))
        else:
            grouped['very_high'].append(max(ber, 1e-6))

    labels = ['very_low', 'low', 'medium', 'high', 'very_high']
    data = [grouped.get(name, []) for name in labels]

    plt.figure(figsize=(7, 4))
    b = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch in b['boxes']:
        patch.set_facecolor('#d3d3d3')

    plt.yscale('log')
    # Match ranges/style from provided reference: set log-scale limits and rotate xlabels
    plt.ylabel('BER (log scale)')
    plt.xlabel('SNR category')
    plt.title('Figure 10 — BER versus SNR (binned)')
    try:
        plt.ylim(1e-3, 1e-1)
    except Exception:
        pass
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main(argv=None):
    import time
    p = argparse.ArgumentParser()
    p.add_argument('--metrics-url', '-m', default='http://127.0.0.1:5001', help='Base URL of the metrics server (default: http://127.0.0.1:5001)')
    p.add_argument(
        '--outdir',
        '-o',
        default=str(Path(__file__).resolve().parent.parent.parent / 'benchmarks' / 'results'),
        help='Base output directory for benchmark-style figure runs (default: benchmarks/results)',
    )
    p.add_argument('--fig', '-f', nargs='+', type=int, choices=[8, 9, 10, 11], default=[8, 9, 10, 11], help='Which figures to generate')
    p.add_argument('--window', '-w', type=int, default=5000, help='Number of recent events to fetch per poll (default: 5000)')
    p.add_argument('--interval', '-i', type=float, default=2.0, help='Polling interval in seconds (default: 2.0)')
    p.add_argument('--rolling', type=int, default=50, help='Rolling window size for Fig 9')
    args = p.parse_args(argv)

    print(f"Polling metrics server every {args.interval}s... Press Ctrl+C to stop and generate plots.")
    
    all_events = {}
    try:
        while True:
            try:
                events_batch = fetch_events(args.metrics_url, window=args.window)
                for e in events_batch:
                    # Use request_id if available, fallback to timestamp
                    key = e.get('request_id') or str(e.get('timestamp'))
                    all_events[key] = e
            except Exception as e:
                print(f"Warning: Failed to fetch events: {e}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nInterrupted! Generating plots from {len(all_events)} collected events...")
    
    if not all_events:
        print("No events collected. Exiting.")
        return
        
    events = sorted(list(all_events.values()), key=lambda x: float(x.get('timestamp', 0)))

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = _latest_model_name(events)
    run_dir = base_outdir / f'figures_8_9_10_{timestamp}_{model_name}'
    run_dir.mkdir(parents=True, exist_ok=True)

    if 8 in args.fig:
        out = os.path.join(run_dir, 'figure8_throughput_snr.png')
        fig8_throughput_vs_snr(events, out)
        print('Wrote', out)

    if 9 in args.fig:
        out = os.path.join(run_dir, 'figure9_spectral_efficiency.png')
        fig9_spectral_efficiency(events, out)
        print('Wrote', out)

    if 10 in args.fig:
        out = os.path.join(run_dir, 'figure10_ber_snr.png')
        fig10_bler_vs_snr_box(events, out)
        print('Wrote', out)

        # Always generate BLER comparison alongside BER when producing Fig.10
        out11 = os.path.join(run_dir, 'figure11_bler_snr.png')
        fig11_bler_vs_snr_box(events, out11)
        print('Wrote', out11)
    elif 11 in args.fig:
        out = os.path.join(run_dir, 'figure11_bler_snr.png')
        fig11_bler_vs_snr_box(events, out)
        print('Wrote', out)

    print(f'Figures saved in {run_dir}')


if __name__ == '__main__':
    main()
