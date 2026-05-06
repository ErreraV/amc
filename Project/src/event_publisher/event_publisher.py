import threading
import queue
import requests
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


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
            self._q.put_nowait((time.time(), event))
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
                    logger.debug(f"Metrics server responded: {r.status_code} {r.text}")
            except Exception as e:
                logger.debug(f"Failed to POST event to metrics server: {e}")

        session.close()

    def stop(self):
        self._stop.set()
        self._thread.join()
