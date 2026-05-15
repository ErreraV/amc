#!/usr/bin/env python3
"""
NR baseline AMC server (no AI) using same TCP interface as RL and Integrated AMC servers.
"""
import json
import socket
import threading
import time
import logging
from collections import deque
import numpy as np
from typing import Dict, Any
from pathlib import Path

from ..event_publisher.event_publisher import EventPublisher

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    try:
        return json.dumps(obj, **kwargs)
    except Exception:
        return json.dumps(str(obj))


class NRAMCServer:
    """Simple NR standard AMC mapping SNR -> MCS with same interface as RLAMCServer."""

    def __init__(self, host='127.0.0.1', port=9001):
        self.host = host
        self.port = port
        self.flow_states = {}
        self.session_start = time.time()
        self.total_requests = 0
        self.event_publisher = EventPublisher()

        # simple SNR to modulation mapping (NR-style bins)
        # thresholds in dB -> modulation string used across repo
        self.snr_bins = [0, 8, 15, 22]
        self.modulations = ['qam4', 'qam16', 'qam64', 'qam256']

        logger.info(f"NRAMCServer started on {host}:{port}")

    def get_flow_state(self, flow_id: int) -> Dict:
        if flow_id not in self.flow_states:
            self.flow_states[flow_id] = {
                'history': {
                    'snr': deque(maxlen=10),
                    'ber': deque(maxlen=10),
                    'bler': deque(maxlen=10),
                    'throughput': deque(maxlen=10),
                    'modulation': deque(maxlen=10),
                    'success': deque(maxlen=10),
                },
                'packet_count': 0
            }
        return self.flow_states[flow_id]

    def _choose_modulation(self, snr: float) -> str:
        for i, thr in enumerate(self.snr_bins[::-1]):
            if snr >= thr:
                return self.modulations[len(self.snr_bins) - 1 - i]
        return self.modulations[0]

    def _fallback_simulation(self, modulation: str, snr: float, channel_type: str) -> Dict:
        # reuse a lightweight simulation for performance estimates
        snr_linear = 10 ** (snr / 10)
        modulation_rates = {'qam4': 2.0, 'qam16': 4.0, 'qam64': 6.0, 'qam256': 8.0}

        ber = 0.01 * np.exp(-snr_linear / 10.0)
        ber = float(np.clip(ber, 1e-8, 0.5))
        bler = min(ber * 30, 0.8)
        success = bler < 0.5

        base_rate = modulation_rates.get(modulation, 4.0)
        if success:
            throughput = base_rate * (1.0 - bler) * (0.9 + 0.05 * np.random.random())
        else:
            throughput = 0.0

        return {
            'ber': float(ber),
            'bler': float(bler),
            'throughput': float(throughput),
            'success': bool(success),
            'processing_time_ms': 0.5
        }

    def process_modulation_request(self, request: Dict) -> Dict:
        try:
            request_time = time.time()
            snr = float(request.get('snr', 20.0))
            flow_id = int(request.get('flow_id', 1))
            channel_info = request.get('channnr_amc_serverel_info', {})

            flow_state = self.get_flow_state(flow_id)
            channel_type = channel_info.get('type', 'rayleigh')

            modulation = self._choose_modulation(snr)

            sim = self._fallback_simulation(modulation, snr, channel_type)

            flow_state['history']['snr'].append(snr)
            flow_state['history']['modulation'].append(modulation)
            flow_state['history']['ber'].append(sim['ber'])
            flow_state['history']['bler'].append(sim['bler'])
            flow_state['history']['throughput'].append(sim['throughput'])
            flow_state['history']['success'].append(sim['success'])
            flow_state['packet_count'] += 1

            response = {
                'modulation': modulation,
                'method': 'NR-Standard',
                'validation': 'simulated',
                'predicted_performance': {
                    'ber': sim['ber'],
                    'bler': sim['bler'],
                    'throughput_effective': sim['throughput'],
                    'success': sim['success']
                },
                'performance_summary': {
                    'avg_throughput': float(np.mean(flow_state['history']['throughput'])) if flow_state['history']['throughput'] else 0.0,
                    'avg_ber': float(np.mean(flow_state['history']['ber'])) if flow_state['history']['ber'] else 0.0,
                    'avg_bler': float(np.mean(flow_state['history']['bler'])) if flow_state['history']['bler'] else 0.0,
                }
            }

            # Publish event to metrics server with same keys used elsewhere
            event = {
                'run_id': getattr(self, '_run_id', 'run-default'),
                'config_id': getattr(self, '_config_id', 'cfg-nr-standard'),
                'model_mode': 'nr_standard',
                'model_name': 'NR Standard',
                'timestamp': time.time(),
                'flow_id': flow_id,
                'request_id': f"{flow_id}-{flow_state['packet_count']}",
                'snr': snr,
                'channel_info': channel_info,
                'chosen_modulation': modulation,
                'decision_method': 'lookup',
                'safety_validated': True,
                'prediction_error': 0.0,
                'decision_latency_ms': (time.time() - request_time) * 1000,
                'sionna_latency_ms': 0.0,
                'gan_latency_ms': 0.0,
                'ber': sim['ber'],
                'bler': sim['bler'],
                'throughput': sim['throughput'],
                'success': sim['success'],
            }
            self.event_publisher.publish(event)

            return response

        except Exception as e:
            logger.error(f"[NR-AMC] Request processing error: {e}")
            return {'modulation': 'qam4', 'method': 'NR-Standard (error)', 'error': str(e)}

    def handle_client(self, conn, addr):
        try:
            data = conn.recv(4096).decode()
            request = json.loads(data)

            request_type = request.get('type', 'get_modulation')
            self.total_requests += 1

            if request_type == 'get_modulation':
                response = self.process_modulation_request(request)
            elif request_type == 'get_stats':
                response = self._get_stats()
            elif request_type == 'reset_stats':
                self._reset_stats()
                response = {'status': 'stats_reset'}
            else:
                response = {'error': f'Unknown request type: {request_type}'}

            conn.sendall(safe_json_dumps(response, indent=2).encode())
        except Exception as e:
            error_response = {'error': f'NR AMC server error: {e}'}
            try:
                conn.sendall(safe_json_dumps(error_response).encode())
            except:
                pass
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            conn.close()

    def _get_stats(self):
        session_duration = time.time() - self.session_start
        return {
            'session': {
                'duration_seconds': float(session_duration),
                'total_requests': int(self.total_requests),
                'active_flows': int(len(self.flow_states))
            }
        }

    def _reset_stats(self):
        self.flow_states = {}
        self.session_start = time.time()
        self.total_requests = 0

    def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind((self.host, self.port))
            sock.listen(10)
            logger.info(f"NRAMCServer listening on {self.host}:{self.port}")

            while True:
                try:
                    conn, addr = sock.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Connection accept error: {e}")
        except Exception as e:
            logger.error(f"Server startup error: {e}")
        finally:
            sock.close()
            logger.info("NRAMCServer stopped")


def main():
    server = NRAMCServer()
    try:
        server.start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == '__main__':
    main()
