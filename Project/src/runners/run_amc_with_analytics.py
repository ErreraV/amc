#!/usr/bin/env python3
"""
Integrated runner for AMC server + Dashboard with real-time histogram visualization
"""

import sys
import threading
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        from ..amc.integrated_amc_gan import IntegratedAMCServer
        from ..servers.dashboard import run_dashboard
        
        # Create AMC server
        logger.info("Initializing integrated AMC server...")
        server = IntegratedAMCServer(
            host='127.0.0.1',
            port=9001
        )
        logger.info("✓ AMC Server initialized")
        
        # Start dashboard in background thread
        logger.info("Starting dashboard server on port 5000...")
        dashboard_thread = threading.Thread(
            target=lambda: run_dashboard(
                server.analyzer,
                host='0.0.0.0',
                port=5000,
                debug=False
            ),
            daemon=True
        )
        dashboard_thread.start()
        time.sleep(1)  # Give dashboard time to start
        logger.info("✓ Dashboard started")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎯 Services Running:")
        logger.info("")
        logger.info("  📊 Dashboard URL:     http://localhost:5000")
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
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
