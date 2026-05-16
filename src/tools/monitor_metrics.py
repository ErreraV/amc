#!/usr/bin/env python3
"""Monitor the metrics server health endpoint and log event counts."""
import time
import requests
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def monitor(metrics_url: str, interval: float, duration: float | None, out_file: str | None):
    end_time = time.time() + duration if duration and duration > 0 else None
    fh = None
    if out_file:
        fh = open(out_file, 'w')
        fh.write('ts,event_count,total_ingested_events,buffer_capacity\n')

    try:
        while True:
            try:
                resp = requests.get(f'{metrics_url}/health', timeout=1.0)
                data = resp.json()
                ts = time.time()
                logger.info(f"health: event_count={data.get('event_count')} total={data.get('total_ingested_events')}")
                if fh:
                    fh.write(f"{ts},{data.get('event_count')},{data.get('total_ingested_events')},{data.get('buffer_capacity')}\n")
                    fh.flush()
            except Exception as e:
                logger.warning(f"Failed to query metrics server: {e}")

            if end_time and time.time() >= end_time:
                break
            time.sleep(interval)
    finally:
        if fh:
            fh.close()


def main():
    parser = argparse.ArgumentParser(description='Poll metrics server /health and log event counts')
    parser.add_argument('--metrics-url', default='http://127.0.0.1:5001', help='Metrics server base URL')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval (s)')
    parser.add_argument('--duration', type=float, default=0.0, help='Total duration to run (s), 0 for indefinite')
    parser.add_argument('--out', default=None, help='Optional CSV output file')
    args = parser.parse_args()

    monitor(args.metrics_url, args.interval, args.duration if args.duration > 0 else None, args.out)


if __name__ == '__main__':
    main()
