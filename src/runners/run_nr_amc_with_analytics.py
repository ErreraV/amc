#!/usr/bin/env python3
"""
Runner for NR-standard AMC + Metrics server + Dashboard.
"""
import sys
import threading
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("NR Standard AMC Server with Dashboard")
    logger.info("=" * 60)

    try:
        from ..servers.metrics_server import run_metrics_server
        from ..servers.dashboard import run_dashboard
        from ..servers.nr_amc_server import NRAMCServer

        logger.info("Starting metrics server on port 5001...")
        metrics_thread = threading.Thread(
            target=lambda: run_metrics_server(host='127.0.0.1', port=5001),
            daemon=True
        )
        metrics_thread.start()
        time.sleep(1)
        logger.info("Metrics server started")

        logger.info("Starting dashboard on port 5000...")
        dashboard_thread = threading.Thread(
            target=lambda: run_dashboard(
                None,
                host='0.0.0.0',
                port=5000,
                debug=False
            ),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(1)
        logger.info("Dashboard started")

        logger.info("Initializing NR-standard AMC server on port 9001...")
        server = NRAMCServer(host='127.0.0.1', port=9001)
        logger.info("NR-standard AMC server initialized")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Services Running")
        logger.info("  Dashboard URL:    http://localhost:5000")
        logger.info("  Metrics server:   http://127.0.0.1:5001")
        logger.info("  AMC server:       127.0.0.1:9001 (NR-Standard)")
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

    return 0


if __name__ == '__main__':
    sys.exit(main())
