#!/usr/bin/env python3
"""Minimal test of dashboard analyzer initialization."""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dashboard_config():
    """Test that dashboard can receive and use analyzer via Flask config."""
    
    # Import Flask components
    from flask import Flask
    from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
    import numpy as np
    
    # Create analyzer and add test data
    logger.info("Creating test analyzer...")
    analyzer = RealtimeAnalyzer()
    
    for i in range(30):
        metrics = PerformanceMetrics(
            flow_id=i % 5,
            decision_method='test',
            predicted_snr=20 + np.random.randn(),
            actual_snr=20 + np.random.randn(),
            prediction_error=abs(np.random.randn()),
            selected_modulation=np.random.choice(['qam4', 'qam16', 'qam64']),
            ber=1e-5 * (1 + np.random.random()),
            bler=0.1 * (1 + np.random.random()),
            throughput=5.0 + np.random.random(),
            decision_latency_ms=30 + np.random.random() * 5,
            sionna_latency_ms=25 + np.random.random() * 5,
            gan_latency_ms=5 + np.random.random() * 2,
            success=True
        )
        analyzer.record_decision(metrics)
    
    logger.info(f"✓ Analyzer has {len(analyzer.metrics_buffer)} metrics")
    
    # Now import and setup dashboard
    logger.info("Importing dashboard...")
    from src.servers.dashboard import app, set_analyzer
    
    # Set analyzer via function
    logger.info("Setting analyzer in Flask config...")
    set_analyzer(analyzer)
    
    # Verify it's in config
    stored_analyzer = app.config.get('ANALYZER')
    logger.info(f"Analyzer in config: {stored_analyzer is not None}")
    logger.info(f"Same object: {stored_analyzer is analyzer}")
    
    # Test API endpoints using Flask's test client
    logger.info("\nTesting endpoints with test client...")
    with app.test_client() as client:
        # Health endpoint
        resp = client.get('/api/health')
        data = resp.get_json()
        logger.info(f"  /api/health -> analyzer_ready: {data.get('analyzer_ready')}")
        
        if not data.get('analyzer_ready'):
            logger.error("✗ Analyzer not ready - config issue!")
            return False
        
        # Realtime stats
        resp = client.get('/api/realtime-stats?window=50')
        data = resp.get_json()
        if 'error' in data:
            logger.error(f"✗ /api/realtime-stats error: {data['error']}")
            return False
        logger.info(f"  /api/realtime-stats -> {len(data)} keys")
        
        # Latency histograms
        resp = client.get('/api/latency-histograms')
        data = resp.get_json()
        if 'error' in data:
            logger.error(f"✗ /api/latency-histograms error: {data['error']}")
            return False
        logger.info(f"  /api/latency-histograms -> {list(data.keys())}")
        
        # Latency breakdown
        resp = client.get('/api/latency-breakdown')
        data = resp.get_json()
        if 'error' in data:
            logger.error(f"✗ /api/latency-breakdown error: {data['error']}")
            return False
        logger.info(f"  /api/latency-breakdown -> {list(data.keys())}")
    
    logger.info("\n✓ All endpoints responding correctly with analyzer data")
    return True

if __name__ == '__main__':
    success = test_dashboard_config()
    sys.exit(0 if success else 1)
