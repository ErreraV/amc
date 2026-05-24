#!/usr/bin/env python3

import numpy as np
import json
import socket
import threading
import time
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import logging
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import random
from pathlib import Path
from ..tools.event_publisher import EventPublisher

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, deque):
        return [json_serializable(item) for item in list(obj)]
    elif isinstance(obj, defaultdict):
        return {key: json_serializable(value) for key, value in dict(obj).items()}
    else:
        return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(json_serializable(obj), **kwargs)


class AdvancedQLearningAgent:
    def __init__(self, state_dim=7, action_dim=4, learning_rate=0.001,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.actions = {
            0: "qam4",
            1: "qam16",
            2: "qam64",
            3: "qam256"
        }

        self.snr_safety_rules = {
            "qam4":   {"min_snr": -10, "max_snr": 5},
            "qam16":  {"min_snr": 1.5, "max_snr": 12},
            "qam64":  {"min_snr": 8.5, "max_snr": 20},
            "qam256": {"min_snr": 15.5, "max_snr": 50}
        }

        self.confidence_threshold = 0.7
        self.bad_choice_penalty = 0.95

        self.recent_bad_choices = []
        self.max_bad_choices_history = 50

        self.modulation_rates = {
            "qam4":   2.0,
            "qam16":  4.0,
            "qam64":  6.0,
            "qam256": 8.0
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_freq = 100
        self.update_counter = 0

        self.state_scaler = StandardScaler()
        self.states_for_normalization = []

        self.performance_history = {
            'throughput': deque(maxlen=1000),
            'ber': deque(maxlen=1000),
            'bler': deque(maxlen=1000),
            'rewards': deque(maxlen=1000),
            'q_values': deque(maxlen=1000),
            'modulation_usage': defaultdict(int),
            'channel_adaptation': defaultdict(lambda: {
                'throughput': deque(maxlen=100),
                'count': 0,
                'success_rate': deque(maxlen=100)
            }),
            'sionna_responses': deque(maxlen=500),
            'snr_modulation_pairs': deque(maxlen=1000)
        }

        self.episode_count = 0
        self.total_reward = 0

        logger.info(f"AdvancedQLearningAgent initialized - state_dim={state_dim}, action_dim={action_dim}")

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def _is_action_safe(self, action: int, snr: float) -> bool:
        modulation = self.actions[action]
        rules = self.snr_safety_rules[modulation]
        is_safe = rules["min_snr"] <= snr <= rules["max_snr"]
        if not is_safe:
            logger.debug(f"Unsafe action: {modulation} @ SNR={snr:.1f}dB "
                         f"(min={rules['min_snr']}dB)")
        return is_safe

    def _get_safe_actions(self, snr: float) -> list:
        safe_actions = [
            action_id for action_id, modulation in self.actions.items()
            if self._is_action_safe(action_id, snr)
        ]
        if not safe_actions:
            safe_actions = [0]
        return safe_actions

    def _calculate_adaptive_epsilon(self) -> float:
        base_epsilon = self.epsilon
        if len(self.recent_bad_choices) >= 5:
            bad_rate = np.mean(self.recent_bad_choices[-10:])
            if bad_rate > 0.3:
                adaptive_epsilon = base_epsilon * 0.5
                logger.debug(f"Epsilon reduced: {bad_rate:.1%} recent bad choices")
            else:
                adaptive_epsilon = base_epsilon
        else:
            adaptive_epsilon = base_epsilon
        return max(adaptive_epsilon, self.epsilon_min)

    def _record_choice_quality(self, snr: float, action: int, reward: float):
        is_bad_choice = (
            reward < -50 or
            not self._is_action_safe(action, snr) or
            (snr > 25 and action == 0) or
            (snr < 8.5 and action >= 2)
        )
        self.recent_bad_choices.append(1.0 if is_bad_choice else 0.0)
        if len(self.recent_bad_choices) > self.max_bad_choices_history:
            self.recent_bad_choices.pop(0)
        if is_bad_choice:
            logger.warning(f"Bad choice detected: SNR={snr:.1f}dB -> "
                           f"{self.actions[action]} (reward={reward:.1f})")

    def encode_state(self, snr: float, ber: float, bler: float,
                     channel_type: str, mobility: float, distance: float,
                     stability: float) -> np.ndarray:
        channel_mapping = {"awgn": 0, "rayleigh": 1, "rice": 2, "rician": 2, "urban": 3}
        channel_encoded = channel_mapping.get(channel_type, 1)

        state = np.array([
            snr / 30.0,
            -np.log10(max(ber, 1e-8)) / 8.0,
            bler,
            channel_encoded / 3.0,
            np.tanh(mobility / 100.0),
            np.tanh(distance / 1000.0),
            np.clip(stability / 5.0, 0, 1)
        ])
        return state

    def select_action(self, state: np.ndarray, snr: float, training: bool = True) -> int:
        safe_actions = self._get_safe_actions(snr)
        adaptive_epsilon = self._calculate_adaptive_epsilon()

        if training and np.random.random() < adaptive_epsilon:
            action = np.random.choice(safe_actions)
            logger.debug(f"Exploration: action {action} from {safe_actions} (eps={adaptive_epsilon:.3f})")
            return action

        self.q_network.eval()

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()

            masked_q_values = q_values.clone()
            for action_id in range(self.action_dim):
                if action_id not in safe_actions:
                    masked_q_values[action_id] = -1e6

            action = masked_q_values.argmax().item()

        if training:
            self.q_network.train()

        self.performance_history['q_values'].append(float(q_values.max().item()))

        q_values_list = q_values.detach().cpu().numpy().tolist()
        logger.debug(f"Exploitation: action {action}, Q-values={q_values_list}, safe_actions={safe_actions}")

        return action

    def calculate_adaptive_reward(self, throughput: float, ber: float, bler: float,
                                  modulation: str, channel_type: str, snr: float,
                                  success: bool = True, prev_modulation: str = None) -> float:
        modulation_order = {"qam4": 0, "qam16": 1, "qam64": 2, "qam256": 3}
        action_id = modulation_order.get(modulation, 1)

        if not self._is_action_safe(action_id, snr):
            extreme_penalty = -500.0
            logger.error(f"Unsafe choice: {modulation} @ SNR={snr:.1f}dB -> penalty: {extreme_penalty}")
            return extreme_penalty

        return self._calculate_standard_reward(throughput, ber, bler, modulation,
                                               channel_type, snr, success, prev_modulation)

    def _calculate_standard_reward(self, throughput: float, ber: float, bler: float,
                                   modulation: str, channel_type: str, snr: float,
                                   success: bool = True, prev_modulation: str = None) -> float:
        modulation_efficiency = {"qam4": 2, "qam16": 4, "qam64": 6, "qam256": 8}
        mod_bits = modulation_efficiency.get(modulation, 4)

        if success and bler < 0.2:
            base_reward = 50.0
        elif success:
            base_reward = 30.0
        else:
            base_reward = -50.0

        throughput_reward = throughput * 10

        snr_mod_reward = 0

        if snr >= 16:
            if modulation == "qam256":
                snr_mod_reward = 50
            elif modulation == "qam64":
                snr_mod_reward = 25
            elif modulation == "qam16":
                snr_mod_reward = -40
            else:
                snr_mod_reward = -60

        elif snr >= 12:
            if modulation == "qam64":
                snr_mod_reward = 45
            elif modulation == "qam256":
                snr_mod_reward = 25
            elif modulation == "qam16":
                snr_mod_reward = 0
            else:
                snr_mod_reward = -30

        elif snr >= 8:
            if modulation == "qam16":
                snr_mod_reward = 40
            elif modulation == "qam64":
                snr_mod_reward = 20
            elif modulation == "qam4":
                snr_mod_reward = 10
            else:
                snr_mod_reward = -10

        elif snr >= 3:
            if modulation == "qam16":
                snr_mod_reward = 35
            elif modulation == "qam4":
                snr_mod_reward = 20
            else:
                snr_mod_reward = -20

        else:
            if modulation == "qam4":
                snr_mod_reward = 40
            else:
                snr_mod_reward = -30

        performance_bonus = 0
        if success:
            if channel_type == "awgn":
                if snr > 20 and modulation in ["qam64", "qam256"]:
                    performance_bonus = 20
            elif channel_type in ["rayleigh", "urban"]:
                if (snr > 20 and modulation in ["qam256", "qam64"]) or \
                   (snr < 15 and modulation == "qam4"):
                    performance_bonus = 15

            if bler < 0.05:
                performance_bonus += 15
            elif bler < 0.1:
                performance_bonus += 8

        ber_penalty = -10 * ber if ber > 0.1 else 0
        bler_penalty = -20 * bler if bler > 0.3 else 0
        switch_penalty = -2 if (prev_modulation and prev_modulation != modulation) else 0

        spectral_efficiency = mod_bits * (1.0 - bler) if success else 0
        efficiency_bonus = spectral_efficiency * 3

        total_reward = (
            base_reward +
            throughput_reward +
            snr_mod_reward +
            performance_bonus +
            ber_penalty +
            bler_penalty +
            switch_penalty +
            efficiency_bonus
        )

        total_reward = np.clip(total_reward, -200, 200)

        logger.info(f"Reward SNR={snr:.1f}dB, Mod={modulation}: "
                    f"Base={base_reward:.1f}, T={throughput_reward:.1f}, "
                    f"SNR-Mod={snr_mod_reward:.1f}, Perf={performance_bonus:.1f}, "
                    f"BER={ber_penalty:.1f}, BLER={bler_penalty:.1f} "
                    f"-> TOTAL={total_reward:.1f}")

        return float(total_reward)

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool = False, snr: float = 20.0):
        self._record_choice_quality(snr, action, reward)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def get_behavior_analysis(self) -> dict:
        if not self.recent_bad_choices:
            return {"status": "insufficient_data"}

        recent_bad_rate = np.mean(self.recent_bad_choices[-20:]) if len(self.recent_bad_choices) >= 20 else 0
        total_bad_rate = np.mean(self.recent_bad_choices)

        return {
            "recent_bad_choice_rate": float(recent_bad_rate),
            "total_bad_choice_rate": float(total_bad_rate),
            "adaptive_epsilon": float(self._calculate_adaptive_epsilon()),
            "base_epsilon": float(self.epsilon),
            "total_choices_recorded": len(self.recent_bad_choices),
            "safety_rules": self.snr_safety_rules,
            "behavior_quality": (
                "GOOD" if recent_bad_rate < 0.1 else
                "MODERATE" if recent_bad_rate < 0.3 else "POOR"
            )
        }

    def update_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.q_network.train()
        self.target_network.eval()

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.discount_factor * next_q_values * ~dones.unsqueeze(1))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug("Target network updated")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        if len(self.states_for_normalization) < 100:
            self.states_for_normalization.append(state)
            if len(self.states_for_normalization) == 100:
                self.state_scaler.fit(self.states_for_normalization)
                logger.info("State normalization initialized")

        if len(self.states_for_normalization) >= 100:
            return self.state_scaler.transform(state.reshape(1, -1))[0]
        else:
            return state

    def update_performance_metrics(self, throughput: float, ber: float, bler: float,
                                   modulation: str, reward: float, channel_type: str,
                                   snr: float, success: bool = True):
        self.performance_history['throughput'].append(throughput)
        self.performance_history['ber'].append(ber)
        self.performance_history['bler'].append(bler)
        self.performance_history['rewards'].append(reward)
        self.performance_history['modulation_usage'][modulation] += 1

        self.performance_history['snr_modulation_pairs'].append({
            'snr': snr,
            'modulation': modulation,
            'throughput': throughput,
            'success': success,
            'bler': bler
        })

        channel_stats = self.performance_history['channel_adaptation'][channel_type]
        channel_stats['throughput'].append(throughput)
        channel_stats['success_rate'].append(1.0 if success else 0.0)
        channel_stats['count'] += 1

        self.total_reward += reward

    def get_performance_summary(self) -> Dict:
        if not self.performance_history['throughput']:
            return {"status": "no_data"}

        recent_rewards = list(self.performance_history['rewards'])[-100:]
        recent_success = [1 for t in list(self.performance_history['throughput'])[-100:] if t > 0]

        snr_mod_analysis = {}
        recent_pairs = list(self.performance_history['snr_modulation_pairs'])[-200:]

        for pair in recent_pairs:
            snr_range = f"{int(pair['snr']/5)*5}-{int(pair['snr']/5)*5+5}dB"
            if snr_range not in snr_mod_analysis:
                snr_mod_analysis[snr_range] = defaultdict(list)
            snr_mod_analysis[snr_range][pair['modulation']].append({
                'throughput': pair['throughput'],
                'success': pair['success']
            })

        return {
            "avg_throughput": float(np.mean(self.performance_history['throughput'])),
            "avg_ber": float(np.mean(self.performance_history['ber'])),
            "avg_bler": float(np.mean(self.performance_history['bler'])),
            "avg_reward": float(np.mean(self.performance_history['rewards'])),
            "recent_avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "success_rate": float(len(recent_success) / len(recent_rewards)) if recent_rewards else 0.0,
            "total_reward": float(self.total_reward),
            "epsilon": float(self.epsilon),
            "episodes": int(self.episode_count),
            "modulation_distribution": dict(self.performance_history['modulation_usage']),
            "recent_q_value": float(np.mean(list(self.performance_history['q_values'])[-10:])) if self.performance_history['q_values'] else 0.0,
            "buffer_size": len(self.replay_buffer),
            "snr_modulation_analysis": {
                snr_range: {
                    mod: {
                        'count': len(data),
                        'avg_throughput': float(np.mean([d['throughput'] for d in data])),
                        'success_rate': float(np.mean([d['success'] for d in data]))
                    }
                    for mod, data in mods.items()
                }
                for snr_range, mods in snr_mod_analysis.items()
            },
            "behavior_analysis": self.get_behavior_analysis()
        }

    def save_model(self, filename: str = "rl_amc_model_safeguard1.pth"):
        # Use models/rl/ directory if relative path is given
        if not Path(filename).is_absolute():
            filename = str(Path(__file__).parent.parent.parent / 'models' / 'rl' / filename)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'recent_bad_choices': self.recent_bad_choices,
            'performance_history': {
                k: (list(v) if isinstance(v, deque) else
                    {k2: list(v2) if isinstance(v2, deque) else v2 for k2, v2 in v.items()}
                    if isinstance(v, defaultdict) else v)
                for k, v in self.performance_history.items()
            },
            'state_scaler': self.state_scaler if len(self.states_for_normalization) >= 100 else None,
            'snr_safety_rules': self.snr_safety_rules
        }, filename)
        logger.info(f"Model saved: {filename}")

    def load_model(self, filename: str = "rl_amc_model_safeguard1.pth") -> bool:
        try:
            # Use models/rl/ directory if relative path is given
            if not Path(filename).is_absolute():
                filename = str(Path(__file__).parent.parent.parent / 'models' / 'rl' / filename)
            
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.episode_count = checkpoint['episode_count']
            self.total_reward = checkpoint['total_reward']

            if 'recent_bad_choices' in checkpoint:
                self.recent_bad_choices = checkpoint['recent_bad_choices']
            if 'snr_safety_rules' in checkpoint:
                self.snr_safety_rules = checkpoint['snr_safety_rules']
            if checkpoint['state_scaler'] is not None:
                self.state_scaler = checkpoint['state_scaler']
                self.states_for_normalization = ['dummy'] * 100

            logger.info(f"Model loaded: {filename}")
            return True
        except FileNotFoundError:
            logger.warning(f"Model file not found: {filename}")
            return False


class RLAMCServer:
    def __init__(self, host='127.0.0.1', port=9001, load_model=True):
        self.host = host
        self.port = port

        self.agent = AdvancedQLearningAgent()
        if load_model:
            self.agent.load_model()

        self.flow_states = {}
        self.session_start = time.time()
        self.total_requests = 0
        self.training_mode = True
        self.event_publisher = EventPublisher()

        self.sionna_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0.0,
            'last_responses': deque(maxlen=100)
        }

        logger.info(f"RLAMCServer started on {host}:{port}")

    def _test_with_sionna(self, modulation: str, snr: float, channel_type: str,
                          channel_info: Dict, flow_id: int) -> Dict:
        start_time = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(('127.0.0.1', 9000))

            k_extended = max(1024, 64)

            np.random.seed(flow_id + int(time.time()) % 1000)
            bits = np.random.randint(0, 2, k_extended).astype(float).tolist()

            payload = {
                "id": flow_id,
                "k": k_extended,
                "modulation": modulation,
                "snr_db": snr,
                "payload": bits,
                "channel_type": channel_type,
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

            sock.sendall(json.dumps(payload).encode())
            response_data = sock.recv(8192)
            sock.close()

            if not response_data:
                raise ConnectionError("Empty response from Sionna")

            response = json.loads(response_data.decode())

            for key in ['success', 'ber', 'bler', 'effective_throughput']:
                if key not in response:
                    logger.warning(f"Missing key in Sionna response: {key}")
                    response[key] = 0.0 if key != 'success' else False

            sionna_throughput = response.get('effective_throughput', 0.0)
            processing_time = time.time() - start_time

            self.sionna_stats['total_calls'] += 1
            if response.get('success', False):
                self.sionna_stats['successful_calls'] += 1
            else:
                self.sionna_stats['failed_calls'] += 1

            total_calls = self.sionna_stats['total_calls']
            self.sionna_stats['avg_response_time'] = (
                self.sionna_stats['avg_response_time'] * (total_calls - 1) + processing_time
            ) / total_calls

            self.sionna_stats['last_responses'].append({
                'flow_id': flow_id,
                'modulation': modulation,
                'snr': snr,
                'ber': response.get('ber', 0),
                'bler': response.get('bler', 0),
                'success': response.get('success', False),
                'processing_time': processing_time,
                'sionna_throughput': sionna_throughput
            })

            return {
                'ber': float(response.get('ber', 0.0)),
                'bler': float(response.get('bler', 0.0)),
                'throughput': float(sionna_throughput),
                'success': bool(response.get('success', False)),
                'processing_time_ms': float(processing_time * 1000),
                'sionna_response': True,
                'extended_bits': k_extended > 64
            }

        except socket.timeout:
            logger.error("Sionna connection timeout")
            self.sionna_stats['failed_calls'] += 1
            return self._fallback_simulation(modulation, snr, channel_type)
        except ConnectionRefusedError:
            logger.error("Sionna connection refused")
            self.sionna_stats['failed_calls'] += 1
            return self._fallback_simulation(modulation, snr, channel_type)
        except json.JSONDecodeError as e:
            logger.error(f"Sionna JSON decode error: {e}")
            self.sionna_stats['failed_calls'] += 1
            return self._fallback_simulation(modulation, snr, channel_type)
        except Exception as e:
            logger.error(f"Sionna error: {e}")
            self.sionna_stats['failed_calls'] += 1
            return self._fallback_simulation(modulation, snr, channel_type)

    def _fallback_simulation(self, modulation: str, snr: float, channel_type: str) -> Dict:
        snr_linear = 10 ** (snr / 10)
        modulation_order = int(modulation.replace("qam", "")) if "qam" in modulation else 4

        modulation_rates = {
            "qam4": 2.0, "qam16": 4.0, "qam64": 6.0, "qam256": 8.0
        }

        if channel_type == "awgn":
            if modulation_order == 4:
                ber = 0.5 * np.exp(-snr_linear)
            elif modulation_order == 16:
                ber = 0.75 * np.exp(-0.8 * snr_linear)
            else:
                ber = 0.9 * np.exp(-0.6 * snr_linear / np.log2(modulation_order))
        else:
            ber = 0.5 / (1 + 0.8 * snr_linear / modulation_order)

        ber *= (0.8 + 0.4 * np.random.random())
        ber = np.clip(ber, 1e-8, 0.5)

        bler = min(ber * 50, 0.8)
        success = bler < 0.5

        base_rate = modulation_rates.get(modulation, 4.0)

        if success:
            quality_factor = 0.95 if ber < 1e-4 else 0.85
            snr_efficiency = min(1.0, (snr + 10) / 40.0)
            channel_overhead = 0.85
            throughput = (
                base_rate *
                (1.0 - bler) *
                quality_factor *
                snr_efficiency *
                channel_overhead
            )
            throughput *= (0.975 + 0.05 * np.random.random())
        else:
            throughput = 0.0

        return {
            'ber': float(ber),
            'bler': float(bler),
            'throughput': float(throughput),
            'success': bool(success),
            'processing_time_ms': 1.0,
            'sionna_response': False,
            'fallback': True
        }

    def get_flow_state(self, flow_id: int) -> Dict:
        if flow_id not in self.flow_states:
            self.flow_states[flow_id] = {
                'history': {
                    'snr': deque(maxlen=10),
                    'ber': deque(maxlen=10),
                    'bler': deque(maxlen=10),
                    'throughput': deque(maxlen=10),
                    'modulation': deque(maxlen=10),
                    'channel_type': deque(maxlen=10),
                    'success': deque(maxlen=10)
                },
                'last_state': None,
                'last_action': None,
                'packet_count': 0,
                'total_reward': 0.0,
                'success_rate': 0.0
            }
        return self.flow_states[flow_id]

    def process_modulation_request(self, request: Dict) -> Dict:
        try:
            request_time = time.time()
            snr = request.get('snr', 20.0)
            flow_id = request.get('flow_id', 1)
            feedback = request.get('feedback', None)
            channel_info = request.get('channel_info', {})

            flow_state = self.get_flow_state(flow_id)
            channel_type = channel_info.get('type', 'rayleigh')
            mobility = channel_info.get('mobility_speed', 3.0)
            distance = channel_info.get('distance', 100.0)

            logger.info(f"[AMC] Request Flow {flow_id}: SNR={snr:.1f}dB, channel={channel_type}")

            if feedback and flow_state['last_state'] is not None:
                self._process_feedback(flow_id, feedback, channel_info)

            flow_state['history']['snr'].append(snr)
            flow_state['history']['channel_type'].append(channel_type)

            stability = self._calculate_stability(flow_state)

            avg_ber = np.mean(flow_state['history']['ber']) if flow_state['history']['ber'] else 1e-6
            avg_bler = np.mean(flow_state['history']['bler']) if flow_state['history']['bler'] else 0.1

            current_state = self.agent.encode_state(
                snr, avg_ber, avg_bler, channel_type, mobility, distance, stability
            )
            normalized_state = self.agent.normalize_state(current_state)

            action = self.agent.select_action(normalized_state, snr, self.training_mode)
            modulation = self.agent.actions[action]

            sionna_result = self._test_with_sionna(modulation, snr, channel_type, channel_info, flow_id)

            if self.training_mode and flow_state['last_state'] is not None:
                reward = self.agent.calculate_adaptive_reward(
                    sionna_result['throughput'],
                    sionna_result['ber'],
                    sionna_result['bler'],
                    modulation,
                    channel_type,
                    snr,
                    sionna_result['success'],
                    flow_state['history']['modulation'][-1] if flow_state['history']['modulation'] else None
                )

                self.agent.store_experience(
                    flow_state['last_state'],
                    flow_state['last_action'],
                    reward,
                    normalized_state,
                    False,
                    snr
                )

                self.agent.update_network()

                self.agent.update_performance_metrics(
                    sionna_result['throughput'],
                    sionna_result['ber'],
                    sionna_result['bler'],
                    modulation,
                    reward,
                    channel_type,
                    snr,
                    sionna_result['success']
                )

                flow_state['total_reward'] += reward

            flow_state['last_state'] = normalized_state
            flow_state['last_action'] = action
            flow_state['packet_count'] += 1

            flow_state['history']['modulation'].append(modulation)
            flow_state['history']['ber'].append(sionna_result['ber'])
            flow_state['history']['bler'].append(sionna_result['bler'])
            flow_state['history']['throughput'].append(sionna_result['throughput'])
            flow_state['history']['success'].append(sionna_result['success'])

            recent_success = list(flow_state['history']['success'])
            flow_state['success_rate'] = np.mean(recent_success) if recent_success else 0.0

            response = {
                'modulation': modulation,
                'method': 'RL-AMC',
                'validation': 'sionna_tested' if sionna_result.get('sionna_response', False) else 'simulated',
                'predicted_performance': {
                    'ber': sionna_result['ber'],
                    'bler': sionna_result['bler'],
                    'throughput_effective': sionna_result['throughput'],
                    'success': sionna_result['success']
                },
                'agent_info': {
                    'epsilon': float(self.agent.epsilon),
                    'adaptive_epsilon': float(self.agent._calculate_adaptive_epsilon()),
                    'action': int(action),
                    'q_value': float(self.agent.performance_history['q_values'][-1]) if self.agent.performance_history['q_values'] else 0.0,
                    'flow_packets': int(flow_state['packet_count']),
                    'flow_reward': float(flow_state['total_reward']),
                    'success_rate': float(flow_state['success_rate']),
                    'snr_context': f"SNR {snr:.1f}dB -> {modulation}",
                    'network_mode': 'eval' if not self.training_mode else 'train'
                },
                'safety_info': {
                    'action_was_safe': self.agent._is_action_safe(action, snr),
                    'safe_actions_available': self.agent._get_safe_actions(snr),
                    'safety_rules_applied': True,
                    'behavior_analysis': self.agent.get_behavior_analysis()
                },
                'sionna_info': {
                    'connected': sionna_result.get('sionna_response', False),
                    'processing_time_ms': float(sionna_result.get('processing_time_ms', 0)),
                    'extended_bits': sionna_result.get('extended_bits', False)
                },
                'performance_summary': self.agent.get_performance_summary()
            }

            logger.info(f"[AMC] Decision: {modulation} | T={sionna_result['throughput']:.2f}Mbps | "
                        f"BER={sionna_result['ber']:.2e} | BLER={sionna_result['bler']:.3f} | "
                        f"Success={sionna_result['success']} | Safe={response['safety_info']['action_was_safe']}")

            # Publish event to metrics server
            event = {
                'run_id': getattr(self, '_run_id', 'run-default'),
                'config_id': getattr(self, '_config_id', 'cfg-rl-only'),
                'model_mode': 'rl_only',
                'model_name': 'RL AMC',
                'timestamp': time.time(),
                'flow_id': flow_id,
                'request_id': f"{flow_id}-{flow_state['packet_count']}",
                'snr': snr,
                'channel_info': channel_info,
                'chosen_modulation': modulation,
                'decision_method': 'rl',
                'safety_validated': response['safety_info']['action_was_safe'],
                'prediction_error': 0.0,
                'decision_latency_ms': (time.time() - request_time) * 1000 if 'request_time' in locals() else 0.0,
                'sionna_latency_ms': sionna_result.get('processing_time_ms', 0.0),
                'gan_latency_ms': 0.0,
                'ber': sionna_result['ber'],
                'bler': sionna_result['bler'],
                'throughput': sionna_result['throughput'],
                'success': sionna_result['success'],
            }
            self.event_publisher.publish(event)

            return response

        except Exception as e:
            logger.error(f"[AMC] Request processing error: {e}")
            return {
                'modulation': 'qam4',
                'method': 'RL-AMC (fallback)',
                'error': str(e),
                'predicted_performance': {
                    'ber': 1e-3,
                    'bler': 0.1,
                    'throughput_effective': 2.0,
                    'success': False
                },
                'safety_info': {
                    'fallback_safe_choice': True,
                    'error_protection': True
                }
            }

    def _process_feedback(self, flow_id: int, feedback: Dict, channel_info: Dict):
        flow_state = self.flow_states[flow_id]

        throughput = feedback.get('throughput', 0.0)
        ber = feedback.get('ber', 0.0)
        bler = feedback.get('bler', 0.0)
        modulation = feedback.get('modulation', 'unknown')

        if ber < 0 or ber > 1:
            logger.warning(f"Invalid BER: {ber}, clamping")
            ber = np.clip(ber, 0, 1)

        if bler < 0 or bler > 1:
            logger.warning(f"Invalid BLER: {bler}, clamping")
            bler = np.clip(bler, 0, 1)

        flow_state['history']['throughput'].append(max(0, throughput))
        flow_state['history']['ber'].append(ber)
        flow_state['history']['bler'].append(bler)

        logger.debug(f"[AMC] Feedback Flow {flow_id}: T={throughput:.2f} | "
                     f"BER={ber:.2e} | BLER={bler:.3f} | Mod={modulation}")

    def _calculate_stability(self, flow_state: Dict) -> float:
        if len(flow_state['history']['modulation']) < 3:
            return 0.0

        recent_mods = list(flow_state['history']['modulation'])[-3:]
        recent_success = list(flow_state['history']['success'])[-3:] if flow_state['history']['success'] else [True] * 3

        unique_mods = len(set(recent_mods))
        mod_stability = (3 - unique_mods) / 2.0

        success_rate = np.mean(recent_success) if recent_success else 0.5
        perf_stability = success_rate

        return (mod_stability + perf_stability) / 2.0

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
            self.total_requests += 1

            if request_type == 'get_modulation':
                response = self.process_modulation_request(request)
            elif request_type == 'get_stats':
                response = self._get_detailed_stats()
            elif request_type == 'get_sionna_stats':
                response = self._get_sionna_stats()
            elif request_type == 'get_behavior_analysis':
                response = self.agent.get_behavior_analysis()
            elif request_type == 'save_model':
                filename = request.get('filename', 'rl_amc_model_safeguard1.pth')
                self.agent.save_model(filename)
                response = {'status': 'model_saved', 'filename': filename}
            elif request_type == 'set_training':
                self.training_mode = request.get('enabled', True)
                response = {'status': 'training_updated', 'training': self.training_mode}
            elif request_type == 'reset_stats':
                self._reset_stats()
                response = {'status': 'stats_reset'}
            else:
                response = {'error': f'Unknown request type: {request_type}'}

            conn.sendall(safe_json_dumps(response, indent=2).encode())
        except Exception as e:
            error_response = {'error': f'AMC server error: {e}'}
            try:
                conn.sendall(safe_json_dumps(error_response).encode())
            except:
                pass
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            conn.close()

    def _get_sionna_stats(self) -> Dict:
        total_calls = max(1, self.sionna_stats['total_calls'])

        return {
            'sionna_connection': {
                'total_calls': int(self.sionna_stats['total_calls']),
                'successful_calls': int(self.sionna_stats['successful_calls']),
                'failed_calls': int(self.sionna_stats['failed_calls']),
                'success_rate': float(self.sionna_stats['successful_calls'] / total_calls),
                'avg_response_time_ms': float(self.sionna_stats['avg_response_time'] * 1000)
            },
            'recent_responses': [
                {
                    'flow_id': int(resp['flow_id']),
                    'modulation': resp['modulation'],
                    'snr': float(resp['snr']),
                    'ber': float(resp['ber']),
                    'bler': float(resp['bler']),
                    'success': bool(resp['success']),
                    'processing_time': float(resp['processing_time']),
                    'sionna_throughput': float(resp.get('sionna_throughput', 0))
                }
                for resp in list(self.sionna_stats['last_responses'])[-10:]
            ],
            'performance_by_modulation': self._analyze_performance_by_modulation()
        }

    def _analyze_performance_by_modulation(self) -> Dict:
        analysis = {}

        for response in self.sionna_stats['last_responses']:
            mod = response['modulation']
            if mod not in analysis:
                analysis[mod] = {
                    'count': 0,
                    'success_rate': 0.0,
                    'avg_ber': 0.0,
                    'avg_bler': 0.0,
                    'avg_processing_time': 0.0,
                    'avg_effective_throughput': 0.0
                }

            stats = analysis[mod]
            count = stats['count']

            stats['success_rate'] = float((stats['success_rate'] * count + (1 if response['success'] else 0)) / (count + 1))
            stats['avg_ber'] = float((stats['avg_ber'] * count + response['ber']) / (count + 1))
            stats['avg_bler'] = float((stats['avg_bler'] * count + response['bler']) / (count + 1))
            stats['avg_processing_time'] = float((stats['avg_processing_time'] * count + response['processing_time']) / (count + 1))

            if 'sionna_throughput' in response:
                stats['avg_effective_throughput'] = float((stats['avg_effective_throughput'] * count + response['sionna_throughput']) / (count + 1))

            stats['count'] += 1

        return analysis

    def _get_detailed_stats(self) -> Dict:
        session_duration = time.time() - self.session_start

        return {
            'session': {
                'duration_seconds': float(session_duration),
                'total_requests': int(self.total_requests),
                'training_mode': bool(self.training_mode),
                'active_flows': int(len(self.flow_states)),
                'requests_per_minute': float(self.total_requests / max(1, session_duration / 60))
            },
            'agent_performance': self.agent.get_performance_summary(),
            'sionna_connection': self._get_sionna_stats()['sionna_connection'],
            'flow_details': {
                str(fid): {
                    'packet_count': int(state['packet_count']),
                    'total_reward': float(state['total_reward']),
                    'success_rate': float(state['success_rate']),
                    'avg_snr': float(np.mean(state['history']['snr'])) if state['history']['snr'] else 0.0,
                    'avg_throughput': float(np.mean(state['history']['throughput'])) if state['history']['throughput'] else 0.0,
                    'recent_modulations': list(state['history']['modulation'])[-5:]
                }
                for fid, state in self.flow_states.items()
            },
            'network_info': {
                'replay_buffer_size': int(len(self.agent.replay_buffer)),
                'epsilon': float(self.agent.epsilon),
                'adaptive_epsilon': float(self.agent._calculate_adaptive_epsilon()),
                'episodes': int(self.agent.episode_count),
                'device': str(self.agent.device),
                'training_mode': bool(self.training_mode),
            },
            'behavior_analysis': self.agent.get_behavior_analysis()
        }

    def _reset_stats(self):
        self.sionna_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0.0,
            'last_responses': deque(maxlen=100)
        }
        self.agent.recent_bad_choices = []
        self.session_start = time.time()
        self.total_requests = 0
        logger.info("Stats reset")

    def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind((self.host, self.port))
            sock.listen(10)

            logger.info(f"RLAMCServer listening on {self.host}:{self.port}")

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
            logger.info("Saving model...")
            self.agent.save_model()

            final_stats = self.agent.get_performance_summary()
            sionna_stats = self._get_sionna_stats()
            behavior_analysis = self.agent.get_behavior_analysis()

            logger.info(f"Total requests: {self.total_requests}")
            logger.info(f"Success rate: {final_stats.get('success_rate', 0):.1%}")
            logger.info(f"Avg reward: {final_stats.get('avg_reward', 0):.1f}")
            logger.info(f"Sionna success rate: {sionna_stats['sionna_connection']['success_rate']:.1%}")

            if behavior_analysis.get('status') != 'insufficient_data':
                logger.info(f"Behavior quality: {behavior_analysis.get('behavior_quality', 'N/A')}")
                logger.info(f"Recent bad choice rate: {behavior_analysis.get('recent_bad_choice_rate', 0):.1%}")

            sock.close()
            logger.info("Server stopped")


def main():
    server = RLAMCServer(load_model=True)
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("Server stopped.")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
