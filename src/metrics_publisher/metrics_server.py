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

# Append event
@app.route('/ingest_event', methods=['POST'])
def ingest_event():
    try:
        event = request.get_json(force=True)
        with _events_lock:
            _events.append(event)
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        logger.exception('Failed ingest event')
        return jsonify({'error': str(e)}), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    window = int(request.args.get('window', EVENT_BUFFER_MAX))
    with _events_lock:
        recent = list(_events)[-window:]
    return jsonify(recent), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'event_count': len(_events)}), 200

def run_metrics_server(host='127.0.0.1', port=5001):
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_metrics_server()
