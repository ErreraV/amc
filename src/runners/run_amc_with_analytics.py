"""Integrated runner for AMC server + Dashboard with real-time histogram visualization.

Description:
    Starts an integrated setup with the AMC server, a metrics server and the
    dashboard so you can observe real-time analytics locally.

How to run:
    From the repository root with your virtualenv active:
        python src/amc/src/runners/run_amc_with_analytics.py

    Use command-line flags for optional arguments; run with `--help` for details.
"""
import argparse
import sys
from multiprocessing import Process
import time
import logging
from pathlib import Path

def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_ensure_repo_root_on_path()

from src.amc.amc import IntegratedAMCServer
from src.servers import dashboard, metrics_server

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Integrated AMC Server with metrics server and dashboard",
    )
    parser.add_argument('--no-gan', action='store_true', help='Disable GAN prediction')
    parser.add_argument('--method', choices=['table', 'rl'], default='rl', help='Decision method to use (table or rl)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='AMC server host')
    parser.add_argument('--port', type=int, default=9001, help='AMC server port')
    parser.add_argument('--metrics-host', type=str, default='127.0.0.1', help='Metrics server host')
    parser.add_argument('--metrics-port', type=int, default=5001, help='Metrics server port')
    parser.add_argument('--dashboard-host', type=str, default='0.0.0.0', help='Dashboard host')
    parser.add_argument('--dashboard-port', type=int, default=5000, help='Dashboard port')
    return parser

def main(argv=None):
    """Start AMC server and dashboard together"""

    args = build_parser().parse_args(argv)
    
    logger.info("=" * 60)
    logger.info("AMC Server with Real-Time Histogram Dashboard")
    logger.info("=" * 60)

    training_data_file = str(Path(__file__).parent.parent.parent / 'data' / 'gan_training_data_enhanced.json')
    preprocessing_file = str(Path(__file__).parent.parent.parent / 'data' / 'enhanced_snr_preprocessing.pkl')

    if not Path(training_data_file).exists():
        logger.warning(f"Training data file '{training_data_file}' not found.")
        training_data_file = None

    if not args.no_gan and not Path(preprocessing_file).exists():
        logger.error(f"Preprocessing file '{preprocessing_file}' not found, but GAN is enabled. Exiting.")
        return 1
    
    try:
        processes = []

        use_gan = not args.no_gan

        # Start metrics server in a separate process
        logger.info("Starting metrics server in a separate process...")
        metrics_process = Process(
            target=start_metrics,
            kwargs={'host': args.metrics_host, 'port': args.metrics_port},
            daemon=True,
        )
        metrics_process.start()
        processes.append(metrics_process)
        time.sleep(0.2)
        logger.info("✓ Metrics server started")
        
        # Create AMC server
        logger.info("Initializing AMC server from src.amc.amc...")
        server = IntegratedAMCServer(
            host=args.host,
            port=args.port,
            training_data_file=training_data_file,
            use_gan=use_gan,
            method=args.method,
        )
        logger.info(f"✓ AMC Server initialized with analyzer: {server.analyzer}")
        
        # Start dashboard in a separate process
        metrics_url = f"http://{args.metrics_host}:{args.metrics_port}"
        logger.info(f"Starting dashboard server on port {args.dashboard_port}...")
        logger.info(f"  Pointing dashboard to metrics server: {metrics_url}")
        dashboard_process = Process(
            target=dashboard.run_dashboard,
            kwargs={
                'host': args.dashboard_host,
                'port': args.dashboard_port,
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
        logger.info(f"  📊 Dashboard URL:     http://{args.dashboard_host}:{args.dashboard_port}")
        logger.info(f"  📡 Metrics Server:    http://{args.metrics_host}:{args.metrics_port}")
        logger.info(f"  🔌 AMC Server:        {args.host}:{args.port}")
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