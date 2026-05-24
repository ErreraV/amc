#!/usr/bin/env python3

import numpy as np
import torch
import logging
import socket
import json
import pickle
import threading
import time
import argparse
from typing import Dict, Tuple, Optional
from collections import deque, defaultdict
import sys
from pathlib import Path

# Ensure repo root is on path so absolute imports work when run directly
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.amc.gan_snr_predictor import SNRGan, GANConfig, SNRDataProcessor
from src.servers.amc_server import AdvancedQLearningAgent, json_serializable, safe_json_dumps
from src.amc.performance_analyzer import RealtimeAnalyzer, PerformanceMetrics
from src.tools.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class FlowStateManager:
    def __init__(self):
        self.flow_states = {}
        self.preselected_modulations = {}

    def get_flow_state(self, flow_id: int) -> Dict:
        if flow_id not in self.flow_states:
            self.flow_states[flow_id] = {
                'snr_history': deque(maxlen=10),
                'modulation_history': deque(maxlen=10),
                'feedback_history': deque(maxlen=10),
                'channel_info': {},
                'last_prediction': None,
                'packet_count': 0
            }
        return self.flow_states[flow_id]

    def store_preselected_modulation(self, flow_id: int, modulation: str, predicted_snr: float):
        self.preselected_modulations[flow_id] = {
            'modulation': modulation,
            'predicted_snr': predicted_snr,
            'timestamp': time.time()
        }

    def retrieve_preselected_modulation(self, flow_id: int) -> Optional[Dict]:
        return self.preselected_modulations.get(flow_id)


class ProductionPreprocessor:
    def __init__(self):
        self.scaler_snr = None
        self.scaler_aux = None
        self.history_len = 15
        self.is_loaded = False
        
        # Fast-path scalars to bypass sklearn overhead
        self.snr_mean = 0.0
        self.snr_scale = 1.0
        self.aux_mean = None
        self.aux_scale = None

    def load_training_preprocessing(self, preprocessing_file=None):
        if preprocessing_file is None:
            preprocessing_file = str(Path(__file__).parent.parent.parent / 'data' / 'enhanced_snr_preprocessing.pkl')
        try:
            with open(preprocessing_file, 'rb') as f:
                params = pickle.load(f)
            self.scaler_snr = params['scaler_snr']
            self.scaler_aux = params['scaler_aux']
            self.history_len = params['history_len']
            
            # Extract raw floats/arrays for lightning-fast native arithmetic
            self.snr_mean = float(self.scaler_snr.mean_[0])
            self.snr_scale = float(self.scaler_snr.scale_[0])
            self.aux_mean = self.scaler_aux.mean_
            self.aux_scale = self.scaler_aux.scale_
            
            self.is_loaded = True
            logger.info("Training preprocessing parameters loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load preprocessing: {e}")
            return False

    def normalize_history(self, snr_history):
        if not self.is_loaded:
            raise ValueError("Preprocessing not loaded")
        # Native numpy vectorization instead of sklearn validation
        arr = np.array(snr_history, dtype=np.float32)
        return (arr - self.snr_mean) / self.snr_scale

    def normalize_aux(self, aux_features):
        if not self.is_loaded:
            raise ValueError("Preprocessing not loaded")
        arr = np.array(aux_features, dtype=np.float32)
        return (arr - self.aux_mean) / self.aux_scale

    def denormalize_snr(self, normalized_snr):
        if not self.is_loaded:
            raise ValueError("Preprocessing not loaded")
        
        # Fast tensor unboxing: bypasses .cpu().detach().numpy() entirely
        if isinstance(normalized_snr, torch.Tensor):
            val = normalized_snr.item() 
        else:
            val = float(normalized_snr)
            
        return (val * self.snr_scale) + self.snr_mean

    @property
    def is_fitted(self):
        return self.is_loaded and self.scaler_snr is not None


class IntegratedAMCServer:
    def __init__(self, host='127.0.0.1', port=9001, training_data_file=None, use_gan=True, method='rl'):
        if training_data_file is None:
            training_data_file = str(Path(__file__).parent.parent.parent / 'data' / 'gan_training_data_enhanced.json')
        self.host = host
        self.port = port
        self.use_gan = use_gan
        self.method = method

        self.flow_manager = FlowStateManager()
        self.preprocessor = ProductionPreprocessor()
        
        if self.use_gan:
            success = self.preprocessor.load_training_preprocessing()
            if not success:
                raise ValueError("Cannot start without proper preprocessing parameters when GAN is enabled")

        # Initialize performance analyzer
        self.analyzer = RealtimeAnalyzer(window_size=500)

        self.rl_agent = AdvancedQLearningAgent(
            state_dim=7,
            action_dim=4,
            learning_rate=0.001
        )
        
        try:
            self.rl_agent.load_model()
            logger.info("Loaded RL agent model")
        except Exception as e:
            logger.warning(f"Could not load existing RL agent model: {e}")
            logger.info("Starting with fresh RL model")

        if self.use_gan:
            gan_config = GANConfig(
                history_len=self.preprocessor.history_len,
                aux_dim=3,
                noise_dim=16,
                ensemble_samples=100
            )
            self.gan = SNRGan(gan_config)
            self.snr_processor = SNRDataProcessor(history_len=self.preprocessor.history_len)

            try:
                enhanced_model_path = str(Path(__file__).parent.parent.parent / 'models' / 'gan' / 'enhanced_snr_gan.pth')
                try:
                    self.gan.load(enhanced_model_path)
                    logger.info(f"Loaded enhanced GAN model: {enhanced_model_path}")
                except FileNotFoundError:
                    logger.warning(f"Enhanced model {enhanced_model_path} not found, trying default...")
                    self.gan.load()
                    logger.info("Loaded default GAN model")
            except Exception as e:
                logger.warning(f"Could not load existing GAN models: {e}")
                logger.info("Starting with fresh GAN models")

        self.sionna_host = '127.0.0.1'
        self.sionna_port = 9000
        self.prediction_threshold = 3.0
        self.event_publisher = EventPublisher()
        
        # Lock to serialize GPU access and prevent GIL thrashing
        self.inference_lock = threading.Lock()

        self.stats = {
            'total_requests': 0,
            'predictions_used': 0,
            'predictions_accurate': 0,
            'fallback_used': 0,
            'denormalization_errors': 0,
            'model_used': str(Path(__file__).parent.parent.parent / 'models' / 'gan' / 'enhanced_snr_gan.pth') if self.use_gan else "None"
        }

        gan_str = "GAN+" if self.use_gan else ""
        logger.info(f"Integrated AMC Server ({gan_str}{self.method.upper()}) initialized on {host}:{port}")
        if self.use_gan:
            logger.info(f"SNR Preprocessor ready: {self.preprocessor.is_fitted}")

    def handle_client(self, conn, addr):
        try:
            data = ""
            while True:
                chunk = conn.recv(65536).decode('utf-8', errors='ignore')
                if not chunk:
                    break
                data += chunk
                try:
                    request = json.loads(data)
                    break
                except json.JSONDecodeError:
                    continue

            if not data:
                return

            request_type = request.get('type', 'get_modulation')

            if request_type == 'get_modulation':
                response = self.process_modulation_request_with_prediction(request)
            elif request_type in ['feedback', 'feedback_only']:
                response = self.handle_feedback(request)
            elif request_type == 'get_stats':
                response = self.get_server_stats()
            else:
                response = {'error': f'Unknown request type: {request_type}'}

            conn.sendall(safe_json_dumps(response, indent=2).encode())
        except Exception as e:
            error_response = {'error': f'Server error: {str(e)}'}
            try:
                conn.sendall(safe_json_dumps(error_response).encode())
            except:
                pass
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            conn.close()

    def process_modulation_request_with_prediction(self, request: Dict) -> Dict:
        try:
            start_time = time.time()
            self.stats['total_requests'] += 1

            flow_id = request.get('flow_id', 1)
            current_snr = request.get('snr', 20.0)
            channel_info = request.get('channel_info', {})

            flow_state = self.flow_manager.get_flow_state(flow_id)
            flow_state['snr_history'].append(current_snr)
            flow_state['channel_info'] = channel_info
            flow_state['packet_count'] += 1

            if self.use_gan:
                self.snr_processor.update_flow_data(
                    flow_id=flow_id,
                    snr=current_snr,
                    mobility_speed=channel_info.get('mobility_speed', 3.0),
                    distance=channel_info.get('distance', 100.0),
                    channel_stability=channel_info.get('channel_stability', 0.5)
                )

                preselected = self.flow_manager.retrieve_preselected_modulation(flow_id)
            else:
                preselected = None

            if preselected and self._is_prediction_still_valid(preselected, current_snr):
                # The prediction was valid, use the proactively preselected modulation!
                selected_modulation = preselected['modulation']
                decision_method = f'preselected_proactive_{self.method}_with_gan'
                prediction_info = {
                    'used_preselected': True,
                    'predicted_snr': preselected['predicted_snr'],
                    'actual_snr': current_snr,
                    'prediction_error': abs(current_snr - preselected['predicted_snr'])
                }
                self.stats['predictions_used'] += 1
                self.stats['predictions_accurate'] += 1
            else:
                # The prediction failed, or GAN is disabled. Use reactive approach on the current SNR.
                selected_modulation = self._select_modulation_current_snr(flow_id, current_snr, channel_info)
                decision_method = f'{self.method}_reactive_fallback' if self.use_gan else f'{self.method}_reactive'
                prediction_info = {'used_preselected': False}
                self.stats['fallback_used'] += 1

            # Trigger the GAN proactive prediction for the NEXT packet (if enabled)
            gan_latency_ms = 0.0
            if self.use_gan:
                gan_start_time = time.time()
                self._proactive_prediction_for_next_transmission(flow_id, channel_info)
                gan_latency_ms = (time.time() - gan_start_time) * 1000

            sionna_start = time.time()
            transmission_result = self._simulate_transmission(selected_modulation, current_snr, channel_info, flow_id)
            sionna_latency_ms = (time.time() - sionna_start) * 1000

            self._handle_transmission_feedback(flow_id, transmission_result, selected_modulation, current_snr)

            flow_state['modulation_history'].append(selected_modulation)
            flow_state['feedback_history'].append(transmission_result)

            decision_latency_ms = (time.time() - start_time) * 1000

            prediction_error = 0.0
            if preselected:
                prediction_error = abs(current_snr - preselected['predicted_snr'])
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                flow_id=flow_id,
                decision_method=decision_method,
                predicted_snr=preselected['predicted_snr'] if preselected else current_snr,
                actual_snr=current_snr,
                prediction_error=prediction_error,
                selected_modulation=selected_modulation,
                ber=transmission_result.get('ber', 0.0),
                bler=transmission_result.get('bler', 0.0),
                throughput=transmission_result.get('throughput', 0.0),
                decision_latency_ms=decision_latency_ms,
                sionna_latency_ms=sionna_latency_ms,
                gan_latency_ms=gan_latency_ms,
                success=transmission_result.get('success', False)
            )
            self.analyzer.record_decision(metrics)

            gan_method_str = 'integrated_enhanced_gan_amc_' if self.use_gan else 'integrated_amc_'
            response = {
                'modulation': selected_modulation,
                'method': f'{gan_method_str}{decision_method}',
                'flow_id': flow_id,
                'decision_info': {
                    'method': decision_method,
                    'prediction_info': prediction_info,
                    'current_snr': current_snr,
                    'safety_validated': self._validate_modulation_safety(selected_modulation, current_snr),
                    'preprocessor_status': self.preprocessor.is_fitted if self.use_gan else False
                },
                'transmission_result': transmission_result,
                'proactive_prediction': {
                    'next_prediction_initiated': self.use_gan,
                    'flow_state_updated': True
                },
                'performance': {
                    'decision_latency_ms': decision_latency_ms,
                    'sionna_latency_ms': sionna_latency_ms,
                    'gan_latency_ms': gan_latency_ms
                }
            }

            model_mode = f"integrated_{'gan_' if self.use_gan else ''}{self.method}"
            model_name = f"Integrated AMC ({'GAN+' if self.use_gan else ''}{self.method.upper()})"
            event = {
                'run_id': getattr(self, '_run_id', 'run-default'),
                'config_id': getattr(self, '_config_id', 'cfg-integrated'),
                'model_mode': model_mode,
                'model_name': model_name,
                'timestamp': time.time(),
                'flow_id': flow_id,
                'request_id': f"{flow_id}-{int(flow_state['packet_count'])}",
                'snr': current_snr,
                'channel_info': channel_info,
                'chosen_modulation': selected_modulation,
                'decision_method': decision_method,
                'safety_validated': self._validate_modulation_safety(selected_modulation, current_snr),
                'prediction_error': prediction_error,
                'decision_latency_ms': decision_latency_ms,
                'sionna_latency_ms': sionna_latency_ms,
                'gan_latency_ms': gan_latency_ms,
                'transmission_source': transmission_result.get('source', 'unknown'),
                'ber': transmission_result.get('ber', 0.0),
                'bler': transmission_result.get('bler', 0.0),
                'throughput': transmission_result.get('throughput', 0.0),
                'success': transmission_result.get('success', False),
            }
            self.event_publisher.publish(event)

            return response
        except Exception as e:
            logger.error(f"Error in integrated processing: {e}")
            return self._emergency_fallback(
                flow_id if 'flow_id' in locals() else 1,
                current_snr if 'current_snr' in locals() else 20.0
            )

    def _proactive_prediction_for_next_transmission(self, flow_id: int, channel_info: Dict):
        try:
            if not self.snr_processor.can_predict(flow_id):
                return

            history, aux = self.snr_processor.prepare_prediction_data(flow_id)
            
            # 1. GAN Predicts Future SNR
            with self.inference_lock, torch.inference_mode():
                raw_predicted_snr = self.gan.predict_next_snr(history, aux, return_confidence=False)

            try:
                predicted_snr_value = self.preprocessor.denormalize_snr(raw_predicted_snr)
            except Exception as e:
                logger.error(f"Denormalization failed: {e}")
                self.stats['denormalization_errors'] += 1
                normalized_val = float(raw_predicted_snr.item())
                predicted_snr_value = 20.0 + normalized_val * 8.0

            if predicted_snr_value < -15.0 or predicted_snr_value > 65.0:
                predicted_snr_value = float(np.clip(predicted_snr_value, -15.0, 65.0))

            logger.info(f"[GAN Prediction] Flow {flow_id} predicted future SNR: {predicted_snr_value:.2f} dB")

            # 2. Make a proactive decision on future SNR using Table or RL
            if self.method == 'table':
                preselected_modulation = self._get_optimal_modulation_for_snr(predicted_snr_value)
            else: # RL
                flow_state = self.flow_manager.get_flow_state(flow_id)
                avg_ber = np.mean([fb.get('ber', 1e-6) for fb in flow_state['feedback_history']]) if flow_state['feedback_history'] else 1e-6
                avg_bler = np.mean([fb.get('bler', 0.1) for fb in flow_state['feedback_history']]) if flow_state['feedback_history'] else 0.1

                state = self.rl_agent.encode_state(
                    snr=predicted_snr_value,
                    ber=avg_ber,
                    bler=avg_bler,
                    channel_type=channel_info.get('type', 'rayleigh'),
                    mobility=channel_info.get('mobility_speed', 3.0),
                    distance=channel_info.get('distance', 100.0),
                    stability=0.5
                )
                normalized_state = self.rl_agent.normalize_state(state)
                
                with self.inference_lock, torch.inference_mode():
                    action_idx = self.rl_agent.select_action(normalized_state, predicted_snr_value, training=False)
                    preselected_modulation = self.rl_agent.actions[action_idx]

            # 3. Store both the predicted SNR and the chosen modulation
            if self._validate_modulation_safety(preselected_modulation, predicted_snr_value):
                logger.info(f"[Proactive Decision] Flow {flow_id} preselected modulation {preselected_modulation} (Method: {self.method.upper()})")
                self.flow_manager.store_preselected_modulation(
                    flow_id, preselected_modulation, predicted_snr_value
                )
            else:
                logger.info(f"[Proactive Decision] Flow {flow_id} preselected modulation {preselected_modulation} failed safety check.")
                               
        except Exception as e:
            logger.error(f"Error in proactive prediction for flow {flow_id}: {e}")

    def _is_prediction_still_valid(self, preselected: Dict, actual_snr: float) -> bool:
        predicted_snr = preselected['predicted_snr']
        prediction_error = abs(actual_snr - predicted_snr)
        is_accurate = prediction_error <= self.prediction_threshold
        modulation = preselected['modulation']
        is_safe = self._validate_modulation_safety(modulation, actual_snr)
        
        if is_accurate and is_safe:
            logger.info(f"[Validation] Prediction valid. Error: {prediction_error:.2f} dB <= {self.prediction_threshold} dB. Modulation {modulation} is safe.")
        else:
            logger.info(f"[Validation] Prediction invalid. Error: {prediction_error:.2f} dB (Accurate: {is_accurate}, Safe: {is_safe}).")
            
        return is_accurate and is_safe

    def _select_modulation_current_snr(self, flow_id: int, snr: float, channel_info: Dict) -> str:
        if self.method == 'table':
            mod = self._get_optimal_modulation_for_snr(snr)
            logger.info(f"[Reactive Decision] Flow {flow_id} selected {mod} using TABLE at SNR {snr:.2f} dB")
            return mod
        else: # RL
            flow_state = self.flow_manager.get_flow_state(flow_id)
            avg_ber = np.mean([fb.get('ber', 1e-6) for fb in flow_state['feedback_history']]) if flow_state['feedback_history'] else 1e-6
            avg_bler = np.mean([fb.get('bler', 0.1) for fb in flow_state['feedback_history']]) if flow_state['feedback_history'] else 0.1

            state = self.rl_agent.encode_state(
                snr=snr,
                ber=avg_ber,
                bler=avg_bler,
                channel_type=channel_info.get('type', 'rayleigh'),
                mobility=channel_info.get('mobility_speed', 3.0),
                distance=channel_info.get('distance', 100.0),
                stability=0.5
            )
            normalized_state = self.rl_agent.normalize_state(state)
            
            with self.inference_lock, torch.inference_mode():
                action = self.rl_agent.select_action(normalized_state, snr, training=False)
                
            mod = self.rl_agent.actions[action]
            logger.info(f"[Reactive Decision] Flow {flow_id} selected {mod} using RL at SNR {snr:.2f} dB")
            return mod

    def _get_optimal_modulation_for_snr(self, snr: float) -> str:
        if snr >= 15.5:
            return 'qam256'
        elif snr >= 8.5:
            return 'qam64'
        elif snr >= 1.5:
            return 'qam16'
        else:
            return 'qam4'

    def _validate_modulation_safety(self, modulation: str, snr: float) -> bool:
        action_mapping = {'qam4': 0, 'qam16': 1, 'qam64': 2, 'qam256': 3}
        action_id = action_mapping.get(modulation, 0)
        return self.rl_agent._is_action_safe(action_id, snr)

    def _simulate_transmission(self, modulation: str, snr: float, channel_info: Dict, flow_id: int) -> Dict:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self.sionna_host, self.sionna_port))

            sionna_request = {
                "type": "amc_server",
                "id": flow_id,
                "k": 1024,
                "modulation": modulation,
                "snr_db": snr,
                "payload": np.random.randint(0, 2, 1024).tolist(),
                "channel_type": channel_info.get('type', 'rayleigh'),
                "channel_params": {
                    "carrier_frequency": channel_info.get('carrier_frequency', 2.4e9),
                    "speed": channel_info.get('mobility_speed', 3.0),
                    "distance": channel_info.get('distance', 100.0),
                    "num_paths": 6,
                    "delay_spread": 1e-6,
                    "k_factor": 10.0,
                    "phase_noise_std": 0.01
                }
            }

            sock.sendall(json.dumps(sionna_request).encode())
            response_data = sock.recv(8192)
            sock.close()

            sionna_response = json.loads(response_data.decode())
            heuristic_result = self._fallback_simulation(modulation, snr, channel_info.get('type', 'rayleigh'))

            return {
                'ber': float(sionna_response.get('ber', 0.0)),
                'bler': float(sionna_response.get('bler', 0.0)),
                'throughput': float(sionna_response.get('effective_throughput', 0.0)),
                'success': bool(sionna_response.get('success', False)),
                'source': 'sionna',
                'actual_ber': float(sionna_response.get('ber', 0.0)),
                'actual_bler': float(sionna_response.get('bler', 0.0)),
                'actual_throughput': float(sionna_response.get('effective_throughput', 0.0)),
                'heuristic_ber': float(heuristic_result.get('ber', 0.0)),
                'heuristic_bler': float(heuristic_result.get('bler', 0.0)),
                'heuristic_throughput': float(heuristic_result.get('throughput', 0.0)),
            }
        except Exception as e:
            fallback_result = self._fallback_simulation(modulation, snr, channel_info.get('type', 'rayleigh'))
            return {
                **fallback_result,
                'source': 'fallback',
                'actual_ber': float('nan'),
                'actual_bler': float('nan'),
                'actual_throughput': float('nan'),
                'heuristic_ber': float(fallback_result.get('ber', 0.0)),
                'heuristic_bler': float(fallback_result.get('bler', 0.0)),
                'heuristic_throughput': float(fallback_result.get('throughput', 0.0)),
            }

    def _fallback_simulation(self, modulation: str, snr: float, channel_type: str) -> Dict:
        snr_linear = 10 ** (snr / 10)
        modulation_order = int(modulation.replace("qam", "")) if "qam" in modulation else 4

        if channel_type == "awgn":
            ber = 0.5 * np.exp(-snr_linear / modulation_order)
        else:
            ber = 0.5 / (1 + 0.8 * snr_linear / modulation_order)

        ber = np.clip(ber * (0.8 + 0.4 * np.random.random()), 1e-8, 0.5)
        bler = min(ber * 50, 0.8)
        success = bler < 0.5

        base_rates = {"qam4": 2.0, "qam16": 4.0, "qam64": 6.0, "qam256": 8.0}
        base_rate = base_rates.get(modulation, 4.0)

        if success:
            throughput = (base_rate * (1.0 - bler) * 0.85) * 50.0
        else:
            throughput = 0.0

        return {
            'ber': float(ber),
            'bler': float(bler),
            'throughput': float(throughput),
            'success': bool(success),
            'source': 'fallback'
        }

    def _handle_transmission_feedback(self, flow_id: int, transmission_result: Dict,
                                       modulation: str, snr: float):
        # try:
        #     reward = self.rl_agent.calculate_adaptive_reward(
        #         throughput=transmission_result['throughput'],
        #         ber=transmission_result['ber'],
        #         bler=transmission_result['bler'],
        #         modulation=modulation,
        #         channel_type='rayleigh',
        #         snr=snr,
        #         success=transmission_result['success']
        #     )
            
        #     with self.inference_lock:
        #         self.rl_agent.update_performance_metrics(
        #             throughput=transmission_result['throughput'],
        #             ber=transmission_result['ber'],
        #             bler=transmission_result['bler'],
        #             modulation=modulation,
        #             reward=reward,
        #             channel_type='rayleigh',
        #             snr=snr,
        #             success=transmission_result['success']
        #         )
        # except Exception as e:
        #     logger.error(f"Error handling feedback for flow {flow_id}: {e}")
        pass

    def _emergency_fallback(self, flow_id: int, snr: float) -> Dict:
        safe_modulation = 'qam4' if snr < 15 else 'qam16'
        return {
            'modulation': safe_modulation,
            'method': 'emergency_fallback',
            'flow_id': flow_id,
            'error': 'system_error_fallback_used'
        }

    def get_server_stats(self) -> Dict:
        prediction_accuracy = 0.0
        if self.stats['predictions_used'] > 0:
            prediction_accuracy = self.stats['predictions_accurate'] / self.stats['predictions_used']

        return {
            'total_requests': self.stats['total_requests'],
            'predictions_used': self.stats['predictions_used'],
            'predictions_accurate': self.stats['predictions_accurate'],
            'fallback_used': self.stats['fallback_used'],
            'denormalization_errors': self.stats['denormalization_errors'],
            'prediction_accuracy_rate': float(prediction_accuracy),
            'prediction_threshold_db': float(self.prediction_threshold),
            'active_flows': len(self.flow_manager.flow_states),
            'cached_preselections': len(self.flow_manager.preselected_modulations),
            'model_used': self.stats['model_used'],
            'preprocessor_fitted': self.preprocessor.is_fitted if self.use_gan else False,
            'gan_enabled': self.use_gan,
            'method': self.method
        }

    def handle_feedback(self, request: Dict) -> Dict:
        try:
            flow_id = request.get('flow_id', 1)
            
            # Resolve nested "feedback" dictionary if present
            feedback_data = request.get('feedback', {})
            if not isinstance(feedback_data, dict) or not feedback_data:
                feedback_data = request
                
            actual_ber = feedback_data.get('ber', 0.0)
            actual_bler = feedback_data.get('bler', 0.0)
            actual_throughput = feedback_data.get('throughput', 0.0)
            modulation_used = feedback_data.get('modulation', 'qam16')
            
            # Resolve SNR from root or nested in channel_info
            channel_info = request.get('channel_info', {})
            snr_used = request.get('snr', channel_info.get('snr', 20.0))

            flow_state = self.flow_manager.get_flow_state(flow_id)
            feedback = {
                'ber': actual_ber,
                'bler': actual_bler,
                'throughput': actual_throughput,
                'success': actual_bler < 0.1
            }
            flow_state['feedback_history'].append(feedback)
            self._handle_transmission_feedback(flow_id, feedback, modulation_used, snr_used)

            return {
                'status': 'feedback_processed',
                'flow_id': flow_id,
                'updated': True
            }
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            return {'error': f'Feedback error: {str(e)}'}

    def save_models_with_enhanced_name(self):
        if self.use_gan:
            try:
                enhanced_path = str(Path(__file__).parent.parent.parent / 'models' / 'gan' / 'enhanced_snr_gan.pth')
                self.gan.save(enhanced_path)
                logger.info(f"Enhanced GAN model saved: {enhanced_path}")
            except Exception as e:
                logger.error(f"Error saving GAN model: {e}")
                
        try:
            with self.inference_lock:
                self.rl_agent.save_model()
            logger.info("RL agent model saved")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")

    def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        gan_str = "GAN+" if self.use_gan else ""
        try:
            sock.bind((self.host, self.port))
            sock.listen(10)

            logger.info(f"Integrated AMC Server ({gan_str}{self.method.upper()}) started on {self.host}:{self.port}")
            if self.use_gan:
                logger.info(f"Prediction accuracy threshold: {self.prediction_threshold} dB")
                logger.info(f"Preprocessor status: {'FITTED' if self.preprocessor.is_fitted else 'NOT_LOADED'}")

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
                    logger.error(f"Error accepting connection: {e}")
        except Exception as e:
            logger.error(f"Error starting server: {e}")
        finally:
            logger.info("Saving models...")
            try:
                self.save_models_with_enhanced_name()
                stats = self.get_server_stats()
                logger.info(f"Total Requests: {stats['total_requests']}")
                if self.use_gan:
                    logger.info(f"Predictions Used: {stats['predictions_used']}")
                    logger.info(f"Prediction Accuracy: {stats['prediction_accuracy_rate']:.1%}")
                    logger.info(f"Denormalization Errors: {stats['denormalization_errors']}")
                    logger.info(f"Preprocessor Fitted: {stats['preprocessor_fitted']}")
                logger.info(f"Fallback Used: {stats['fallback_used']}")
            except Exception as e:
                logger.error(f"Error saving models/stats: {e}")
            sock.close()
            logger.info("Integrated AMC Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Integrated AMC Server (GAN / No-GAN, Table / RL)")
    parser.add_argument('--no-gan', action='store_true', help='Disable GAN prediction')
    parser.add_argument('--method', choices=['table', 'rl'], default='rl', help='Decision method to use (table or rl)')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9001)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )

    use_gan = not args.no_gan
    training_data_file = str(Path(__file__).parent.parent.parent / 'data' / 'gan_training_data_enhanced.json')
    preprocessing_file = str(Path(__file__).parent.parent.parent / 'data' / 'enhanced_snr_preprocessing.pkl')

    if not Path(training_data_file).exists():
        logger.warning(f"Training data file '{training_data_file}' not found.")
        training_data_file = None

    if use_gan and not Path(preprocessing_file).exists():
        logger.error(f"Preprocessing file '{preprocessing_file}' not found, but GAN is enabled. Exiting.")
        return

    server = IntegratedAMCServer(
        host=args.host,
        port=args.port,
        training_data_file=training_data_file,
        use_gan=use_gan,
        method=args.method
    )

    try:
        server.start_server()
    except KeyboardInterrupt:
        print("Server stopped.")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()
