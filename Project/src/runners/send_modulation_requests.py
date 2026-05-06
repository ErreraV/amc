#!/usr/bin/env python3
"""Send repeated get_modulation TCP requests to AMC server for testing."""
import socket
import json
import time
import random
import argparse

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
    except Exception:
        pass
    time.sleep(1.0 / max(1.0, args.rate))

print('Sender finished')
