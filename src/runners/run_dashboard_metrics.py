#!/usr/bin/env python3
"""Runner: start metrics server and dashboard (no AMC server).

Run from workspace root with the venv activated:
  python src/amc_/run_dashboard_metrics.py

This script starts the metrics server on port 5001 and the dashboard on port 5000.
"""
from multiprocessing import Process
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


def _stop_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=5)


def start_metrics(host: str = '127.0.0.1', port: int = 5001):
    print(f"Starting metrics server on {host}:{port}")
    metrics_server.run_metrics_server(host=host, port=port)


def main():
    metrics_host = '127.0.0.1'
    metrics_port = 5001
    dashboard_host = '0.0.0.0'
    dashboard_port = 5000

    processes = []

    # Start metrics server in a separate process
    metrics_process = Process(
        target=start_metrics,
        kwargs={'host': metrics_host, 'port': metrics_port},
        daemon=True,
    )
    metrics_process.start()
    processes.append(metrics_process)

    # Give the metrics server a moment to start
    import time
    time.sleep(0.2)

    metrics_url = f"http://{metrics_host}:{metrics_port}"
    print(f"Starting dashboard on {dashboard_host}:{dashboard_port}, pointing to {metrics_url}")
    dashboard_process = Process(
        target=dashboard.run_dashboard,
        kwargs={
            'host': dashboard_host,
            'port': dashboard_port,
            'debug': False,
            'metrics_server_url': metrics_url,
        },
        daemon=True,
    )
    dashboard_process.start()
    processes.append(dashboard_process)

    try:
        dashboard_process.join()
    except KeyboardInterrupt:
        print("Stopping dashboard and metrics processes...")
    finally:
        _stop_processes(processes)


if __name__ == '__main__':
    main()
