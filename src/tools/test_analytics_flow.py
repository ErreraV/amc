#!/usr/bin/env python3
"""
Test script to verify the full analytics flow.
This script simulates requests to the AMC server and monitors the analyzer.
Run this in parallel with run_amc_with_analytics.py to verify data flows to dashboard.
"""

import sys
import socket
import json
import time
import random
import threading
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def send_requests(host='127.0.0.1', port=9001, duration_s=30, rate_hz=5):
    """Send modulation requests to AMC server"""
    logger.info(f"Starting to send requests to {host}:{port}")
    logger.info(f"Duration: {duration_s}s | Rate: {rate_hz} req/s")
    
    end_time = time.time() + duration_s
    request_count = 0
    success_count = 0
    error_count = 0
    
    while time.time() < end_time:
        flow_id = random.randint(1, 5)
        snr = random.uniform(-5, 30)
        
        payload = {
            'type': 'get_modulation',
            'flow_id': flow_id,
            'snr': snr,
            'channel_info': {
                'type': 'rayleigh',
                'mobility_speed': 3.0,
                'distance': 100.0
            }
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.sendall(json.dumps(payload).encode())
            response = sock.recv(8192)
            sock.close()
            success_count += 1
            request_count += 1
            logger.debug(f"Request {request_count}: Flow {flow_id}, SNR {snr:.1f}dB - OK")
        except Exception as e:
            error_count += 1
            logger.warning(f"Request failed: {e}")
        
        sleep_time = 1.0 / max(1.0, rate_hz)
        time.sleep(sleep_time)
    
    logger.info("=" * 60)
    logger.info(f"Request sending complete:")
    logger.info(f"  Total sent:    {request_count}")
    logger.info(f"  Successful:    {success_count}")
    logger.info(f"  Failed:        {error_count}")
    logger.info("=" * 60)


def check_metrics_server(metrics_url='http://127.0.0.1:5001'):
    """Check if metrics server has events"""
    import requests
    
    try:
        response = requests.get(f'{metrics_url}/health', timeout=2)
        data = response.json()
        logger.info(f"✓ Metrics server health: {data}")
        
        # Try to fetch some events
        events = requests.get(f'{metrics_url}/api/events?window=10', timeout=2).json()
        logger.info(f"✓ Metrics server has {len(events)} recent events")
        return True
    except Exception as e:
        logger.error(f"✗ Metrics server check failed: {e}")
        return False


def check_dashboard(dashboard_url='http://127.0.0.1:5000'):
    """Check if dashboard is responding"""
    import requests
    
    try:
        response = requests.get(f'{dashboard_url}/api/health', timeout=2)
        data = response.json()
        logger.info(f"✓ Dashboard health: {data}")
        
        # Try to get realtime stats
        stats = requests.get(f'{dashboard_url}/api/realtime-stats', timeout=2).json()
        if 'error' in stats:
            logger.warning(f"Dashboard returned error: {stats['error']}")
        else:
            logger.info(f"✓ Dashboard stats available: {list(stats.keys())}")
        return True
    except Exception as e:
        logger.error(f"✗ Dashboard check failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test analytics flow: send requests and monitor data'
    )
    parser.add_argument('--amc-host', default='127.0.0.1', help='AMC server host')
    parser.add_argument('--amc-port', type=int, default=9001, help='AMC server port')
    parser.add_argument('--duration', type=float, default=30, help='Duration to send requests (seconds)')
    parser.add_argument('--rate', type=float, default=5, help='Request rate (req/sec)')
    parser.add_argument('--metrics-url', default='http://127.0.0.1:5001', help='Metrics server URL')
    parser.add_argument('--dashboard-url', default='http://127.0.0.1:5000', help='Dashboard URL')
    parser.add_argument('--check-only', action='store_true', help='Only check services, don\'t send requests')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Analytics Flow Test")
    logger.info("=" * 60)
    
    # Check services
    logger.info("\n[CHECKING SERVICES]")
    metrics_ok = check_metrics_server(args.metrics_url)
    dashboard_ok = check_dashboard(args.dashboard_url)
    
    if not metrics_ok or not dashboard_ok:
        logger.error("One or more services are not responding. Make sure run_amc_with_analytics.py is running.")
        return 1
    
    if args.check_only:
        logger.info("Service check complete. Exiting.")
        return 0
    
    logger.info("\n[SENDING REQUESTS]")
    logger.info("Note: Open http://localhost:5000 in your browser to see data appear on the dashboard")
    logger.info("")
    
    # Send requests
    send_requests(
        host=args.amc_host,
        port=args.amc_port,
        duration_s=args.duration,
        rate_hz=args.rate
    )
    
    # Check again
    logger.info("\n[FINAL CHECK]")
    check_metrics_server(args.metrics_url)
    check_dashboard(args.dashboard_url)
    
    logger.info("\n[COMPLETE]")
    logger.info("If you see data in the checks above, the dashboard should be displaying it.")
    logger.info("If not, check:")
    logger.info("  1. AMC server is running and accepting connections")
    logger.info("  2. Dashboard server is running (http://localhost:5000)")
    logger.info("  3. Check browser console for JavaScript errors")
    
    return 0
    

if __name__ == '__main__':
    sys.exit(main())
