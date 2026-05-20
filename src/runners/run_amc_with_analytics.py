#!/usr/bin/env python3
"""
Integrated runner for AMC server + Dashboard with real-time histogram visualization
"""

import sys
from multiprocessing import Process
import time
import logging
from pathlib import Path

if __package__:
    from ..servers import dashboard, metrics_server
else:
    import sys as _sys

    here = Path(__file__).resolve().parent
    src_dir = None
    for p in [here] + list(here.parents):
        if (p / 'servers').is_dir():
            src_dir = p
            break
    if src_dir is None:
        src_dir = here.parent
    _sys.path.insert(0, str(src_dir))
    from servers import dashboard, metrics_server

# Configure logging
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


def start_metrics(host: str = '127.0.0.1', port: int = 5001):
    logger.info(f"Starting metrics server on {host}:{port}")
    metrics_server.run_metrics_server(host=host, port=port)

def main():
    """Start AMC server and dashboard together"""
    
    logger.info("=" * 60)
    logger.info("AMC Server with Real-Time Histogram Dashboard")
    logger.info("=" * 60)
    
    # Check required files (in data directory)
    required_files = [
        Path(__file__).parent.parent.parent / 'data' / 'enhanced_snr_preprocessing.pkl',
    ]
    
    for fname in required_files:
        if not fname.exists():
            logger.error(f"❌ Required file not found: {fname}")
            return 1
    
    logger.info("✓ All required files found")
    
    try:
        # Import after checking files
        from ..amc.integrated_amc_gan_enzo import IntegratedAMCServer

        metrics_host = '127.0.0.1'
        metrics_port = 5001
        dashboard_host = '0.0.0.0'
        dashboard_port = 5000
        processes = []

        # Start metrics server in a separate process
        logger.info("Starting metrics server in a separate process...")
        metrics_process = Process(
            target=start_metrics,
            kwargs={'host': metrics_host, 'port': metrics_port},
            daemon=True,
        )
        metrics_process.start()
        processes.append(metrics_process)
        time.sleep(0.2)
        logger.info("✓ Metrics server started")
        
        # Create AMC server
        logger.info("Initializing integrated AMC server...")
        server = IntegratedAMCServer(
            host='127.0.0.1',
            port=9001
        )
        logger.info(f"✓ AMC Server initialized with analyzer: {server.analyzer}")
        
        # Start dashboard in a separate process
        metrics_url = f"http://{metrics_host}:{metrics_port}"
        logger.info(f"Starting dashboard server on port {dashboard_port}...")
        logger.info(f"  Pointing dashboard to metrics server: {metrics_url}")
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
        time.sleep(1)  # Give dashboard time to start
        logger.info("✓ Dashboard started")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎯 Services Running:")
        logger.info("")
        logger.info("  📊 Dashboard URL:     http://localhost:5000")
        logger.info("  📡 Metrics Server:    http://127.0.0.1:5001")
        logger.info("  🔌 AMC Server:        127.0.0.1:9001")
        logger.info("  📈 Sionna Expected:   127.0.0.1:9000")
        logger.info("")
        logger.info("Features:")
        logger.info("  • Real-time latency histograms")
        logger.info("  • Decision/Sionna/GAN latency breakdown")
        logger.info("  • Flow-level performance metrics")
        logger.info("  • Prediction accuracy tracking")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        logger.info("")
        
        # Start AMC server (blocking)
        server.start_server()
        
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    finally:
        if 'processes' in locals():
            _stop_processes(processes)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())