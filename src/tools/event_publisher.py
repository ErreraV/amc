"""Asynchronous event publisher used by services to record and forward events.

Description:
    Provides `EventPublisher` which queues events, writes JSONL to disk and
    POSTs events to a metrics server in the background.

How to use:
    Import and construct `EventPublisher` in the process that produces events:
        from src.tools.event_publisher import EventPublisher
        pub = EventPublisher(server_url='http://127.0.0.1:5001/ingest_event')

The publisher starts a background thread automatically.
"""

import threading
import queue
import requests
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert nested values to JSON-serializable Python primitives."""
    # numpy/tensor scalar-like objects usually expose item()
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return _json_safe(item_fn())
        except Exception:
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # numpy arrays usually expose tolist()
    tolist_fn = getattr(value, "tolist", None)
    if callable(tolist_fn):
        try:
            return _json_safe(tolist_fn())
        except Exception:
            pass

    return str(value)


class EventPublisher:
    """Asynchronous publisher that writes JSONL to disk and POSTs to metrics server."""

    def __init__(self, server_url: str = "http://127.0.0.1:5001/ingest_event", disk_path: Optional[str] = None, max_queue=10000):
        self.server_url = server_url
        self.disk_path = Path(disk_path) if disk_path else None
        self._q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def publish(self, event: Dict[str, Any]):
        try:
            safe_event = _json_safe(event)
            self._q.put_nowait((time.time(), safe_event))
            try:
                qsize = self._q.qsize()
            except Exception:
                qsize = -1
            logger.info(f"EventPublisher.enqueue: queued event type={safe_event.get('type')} qsize={qsize}")
        except queue.Full:
            logger.warning("EventPublisher queue full, dropping event")

    def _worker(self):
        session = requests.Session()
        headers = {"Content-Type": "application/json"}
        while not self._stop.is_set():
            try:
                ts, event = self._q.get(timeout=0.5)
            except Exception:
                continue

            # log processing
            try:
                qsize = self._q.qsize()
            except Exception:
                qsize = -1
            logger.info(f"EventPublisher.worker: processing event type={event.get('type')} qsize={qsize}")

            # write to disk
            if self.disk_path:
                try:
                    self.disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.disk_path, 'a') as f:
                        f.write(json.dumps(event) + "\n")
                except Exception as e:
                    logger.exception(f"Failed writing event to disk: {e}")

            # post to metrics server
            try:
                r = session.post(self.server_url, json=event, timeout=1.0, headers=headers)
                if r.status_code != 200:
                    logger.warning(f"Metrics server responded: {r.status_code} {r.text}")
            except Exception as e:
                logger.warning(f"Failed to POST event to metrics server: {e}")

        session.close()

    def stop(self):
        self._stop.set()
        self._thread.join()
