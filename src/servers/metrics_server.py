#!/usr/bin/env python3
"""Metrics server: ingest raw events and expose raw event distribution APIs."""
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# In-memory event buffer
EVENT_BUFFER_MAX = 5000
_events = deque(maxlen=EVENT_BUFFER_MAX)
_events_lock = threading.Lock()
_total_ingested_events = 0


@app.route('/ingest_event', methods=['POST'])
def ingest_event():
    global _total_ingested_events
    try:
        event = request.get_json(force=True)
        with _events_lock:
            _events.append(event)
            _total_ingested_events += 1
            if _total_ingested_events % 10 == 0:
                logger.info(f"ingest_event: total_ingested_events={_total_ingested_events} event_type={event.get('type')}")
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        logger.exception('Failed ingest event')
        return jsonify({'error': str(e)}), 500


@app.route('/api/events', methods=['GET'])
def get_events():
    with _events_lock:
        window_arg = request.args.get('window')
        if window_arg is None:
            recent = list(_events)
        else:
            window = int(window_arg)
            if window <= 0:
                recent = list(_events)
            else:
                recent = list(_events)[-window:]
    return jsonify(recent), 200


@app.route('/health', methods=['GET'])
def health():
    with _events_lock:
        event_count = len(_events)
        total_ingested_events = _total_ingested_events
    return jsonify({
        'status': 'ok',
        'event_count': event_count,
        'total_ingested_events': total_ingested_events,
        'buffer_capacity': EVENT_BUFFER_MAX,
    }), 200


def run_metrics_server(host='127.0.0.1', port=5001):
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    run_metrics_server()
