"""Debug script to test analyzer connection in dashboard.

Description:
    Small debug utility that creates a `RealtimeAnalyzer` instance, injects
    synthetic performance metrics and exercises the dashboard's endpoints to
    validate analyzer integration.

How to run:
    From the repository root with your virtualenv active:
        python src/amc/src/tools/debug_analyzer.py

    This is intended as a quick local test (it exits on import failure).
"""

import sys
import time
import threading
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test imports
logger.info("Testing imports...")
try:
    from src.amc.integrated_amc_gan import IntegratedAMCServer
    from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
    from src.servers.metrics_server import run_metrics_server
    from src.servers.dashboard import run_dashboard, app
    logger.info("✓ All imports successful")
except ImportError as e:
    logger.error(f"✗ Import failed: {e}")
    sys.exit(1)

# Create analyzer and test it
logger.info("\nCreating test analyzer...")
analyzer = RealtimeAnalyzer()
logger.info(f"Analyzer created: {analyzer}")
logger.info(f"Analyzer type: {type(analyzer)}")

# Add some test metrics
logger.info("Adding test metrics to analyzer...")
import numpy as np
for i in range(20):
    metrics = PerformanceMetrics(
        flow_id=i % 5,
        decision_method='test',
        predicted_snr=20,
        actual_snr=20,
        prediction_error=0,
        selected_modulation='qam16',
        ber=1e-5,
        bler=0.1,
        throughput=5.0,
        decision_latency_ms=30,
        sionna_latency_ms=25,
        gan_latency_ms=5,
        success=True
    )
    analyzer.record_decision(metrics)

logger.info(f"Analyzer buffer size: {len(analyzer.metrics_buffer)}")
stats = analyzer.get_real_time_stats()
logger.info(f"Stats from analyzer: {list(stats.keys())}")

# Test setting analyzer in Flask config
logger.info("\nTesting Flask config...")
from src.servers.dashboard import set_analyzer
set_analyzer(analyzer)
logger.info(f"Analyzer set in Flask config: {app.config.get('ANALYZER')}")
logger.info(f"Config ANALYZER is same object: {app.config.get('ANALYZER') is analyzer}")

# Test health endpoint
logger.info("\nTesting health endpoint...")
with app.test_client() as client:
    response = client.get('/api/health')
    logger.info(f"Health endpoint response: {response.get_json()}")
    
    response = client.get('/api/realtime-stats')
    data = response.get_json()
    logger.info(f"Stats endpoint response keys: {list(data.keys()) if not 'error' in data else 'ERROR'}")
    if 'error' in data:
        logger.error(f"Stats endpoint error: {data['error']}")

logger.info("\n✓ Debug test complete - analyzer appears to be working correctly")
logger.info("If dashboard still shows 'Analyzer not initialized', the issue is elsewhere.")
