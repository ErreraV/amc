#!/usr/bin/env python3
"""Runner: start metrics server and dashboard (no AMC server).

Run from workspace root with the venv activated:
  python src/amc_/run_dashboard_metrics.py

This script starts the metrics server on port 5001 and the dashboard on port 5000.
"""
import threading
from pathlib import Path

if __package__:
    # Module execution, e.g. `python -m src.runners.run_dashboard_metrics`.
    from ..servers import dashboard, metrics_server
else:
    # Direct script execution fallback.
    import sys

    here = Path(__file__).resolve().parent
    src_dir = None
    for p in [here] + list(here.parents):
        if (p / 'servers').is_dir():
            src_dir = p
            break
    if src_dir is None:
        src_dir = here.parent
    sys.path.insert(0, str(src_dir))
    from servers import dashboard, metrics_server


def start_metrics(host: str = '127.0.0.1', port: int = 5001):
    print(f"Starting metrics server on {host}:{port}")
    metrics_server.run_metrics_server(host=host, port=port)


def main():
    metrics_host = '127.0.0.1'
    metrics_port = 5001
    dashboard_host = '0.0.0.0'
    dashboard_port = 5000

    # Start metrics server in background thread
    t = threading.Thread(
        target=start_metrics,
        kwargs={'host': metrics_host, 'port': metrics_port},
        daemon=True,
    )
    t.start()

    # Give the metrics server a moment to start
    import time
    time.sleep(0.2)

    metrics_url = f"http://{metrics_host}:{metrics_port}"
    print(f"Starting dashboard on {dashboard_host}:{dashboard_port}, pointing to {metrics_url}")
    dashboard.run_dashboard(host=dashboard_host, port=dashboard_port, debug=False, metrics_server_url=metrics_url)


if __name__ == '__main__':
    main()
