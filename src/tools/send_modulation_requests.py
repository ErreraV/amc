#!/usr/bin/env python3
"""Send repeated get_modulation TCP requests to AMC server for testing."""
import socket
import json
import time
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9001)
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--rate', type=float, default=5.0)  # requests per second
    args = parser.parse_args()

    end_time = time.time() + args.duration
    flow_id = 1

    while time.time() < end_time:
        snr = random.uniform(-5, 30)
        payload = {
            'type': 'get_modulation',
            'flow_id': flow_id,
            'snr': snr,
            'channel_info': {'type': 'rayleigh', 'mobility_speed': 3.0, 'distance': 100.0}
        }
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((args.host, args.port))
            sock.sendall(json.dumps(payload).encode())
            data = sock.recv(8192)
            # ignore response
            sock.close()
            logger.debug('Sent request')
        except Exception:
            logger.debug('Request failed')
        time.sleep(1.0 / max(1.0, args.rate))

    logger.info('Sender finished')


if __name__ == '__main__':
    main()
