#!/usr/bin/env python3
"""
Runner for RL-only AMC server + Metrics server + Dashboard.
"""

import sys
from multiprocessing import Process
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _stop_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=5)


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main():
    """Start RL-only AMC, metrics bridge, and dashboard."""

    logger.info("=" * 60)
    logger.info("RL-Only AMC Server with Dashboard")
    logger.info("=" * 60)

    required_files = [
        Path(__file__).parent.parent.parent / 'models' / 'rl' / 'rl_amc_model_safeguard1.pth',
    ]

    for fname in required_files:
        if not fname.exists():
            logger.warning(f"Model file not found (server will start with fresh state): {fname}")

    try:
        _ensure_repo_root_on_path()
        from src.servers.metrics_server import run_metrics_server  # type: ignore
        from src.servers.amc_server import RLAMCServer  # type: ignore
        from src.servers.dashboard import run_dashboard  # type: ignore

        processes = []

        logger.info("Starting metrics server on port 5001...")
        metrics_process = Process(
            target=run_metrics_server,
            kwargs={'host': '127.0.0.1', 'port': 5001},
            daemon=True,
        )
        metrics_process.start()
        processes.append(metrics_process)
        time.sleep(1)
        logger.info("Metrics server started")

        logger.info("Starting dashboard on port 5000...")
        dashboard_process = Process(
            target=run_dashboard,
            kwargs={
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'metrics_server_url': 'http://127.0.0.1:5001',
            },
            daemon=True,
        )
        dashboard_process.start()
        processes.append(dashboard_process)
        time.sleep(1)
        logger.info("Dashboard started")

        logger.info("Initializing RL-only AMC server on port 9001...")
        server = RLAMCServer(
            host='127.0.0.1',
            port=9001,
            load_model=True
        )
        logger.info("RL-only AMC server initialized")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Services Running")
        logger.info("  Dashboard URL:    http://localhost:5000")
        logger.info("  Metrics server:   http://127.0.0.1:5001")
        logger.info("  AMC server:       127.0.0.1:9001 (RL-only)")
        logger.info("  Sionna expected:  127.0.0.1:9000")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        logger.info("")

        server.start_server()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        return 1
    finally:
        if 'processes' in locals():
            _stop_processes(processes)

    return 0


if __name__ == '__main__':
    sys.exit(main())
