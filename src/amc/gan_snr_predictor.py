"""GAN-based SNR predictor: model definitions and helpers.

Description:
    Implements `SNRGan`, generator/discriminator models and `GANConfig` used
    to predict SNR values from historical channel data.

How to use:
    Import the classes where you need them:
        from src.amc.gan_snr_predictor import SNRGan, GANConfig

    Training and evaluation are performed via project-specific training scripts
    (see the `models/gan` folder for saved checkpoints).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import json
import os
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class GANConfig:
    history_len: int = 10
    aux_dim: int = 3
    noise_dim: int = 16

    gen_hidden_dims: List[int] = None
    gen_dropout: float = 0.2
    gen_activation: str = "leaky_relu"

    disc_hidden_dims: List[int] = None
    disc_dropout: float = 0.3
    disc_activation: str = "leaky_relu"

    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    adversarial_weight: float = 1.0
    l1_penalty_weight: float = 10.0
    gradient_penalty_weight: float = 10.0

    snr_min: float = -10.0
    snr_max: float = 35.0

    ensemble_samples: int = 20
    ensemble_method: str = "median"

    device: str = "auto"
    dtype: torch.dtype = torch.float32

    enable_logging: bool = True
    log_frequency: int = 100

    def __post_init__(self):
        if self.gen_hidden_dims is None:
            self.gen_hidden_dims = [128, 256, 128]

        if self.disc_hidden_dims is None:
            self.disc_hidden_dims = [128, 256, 128, 64]

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(self.dtype, str):
            if self.dtype == "float32":
                self.dtype = torch.float32
            elif self.dtype == "float16":
                self.dtype = torch.float16
            elif self.dtype == "float64":
                self.dtype = torch.float64
            else:
                self.dtype = torch.float32

        if self.history_len <= 0:
            raise ValueError("history_len must be > 0")
        if self.aux_dim <= 0:
            raise ValueError("aux_dim must be > 0")
        if self.noise_dim <= 0:
            raise ValueError("noise_dim must be > 0")


class SNRGenerator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        self.input_dim = config.history_len + config.aux_dim + config.noise_dim

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in config.gen_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(config.gen_activation),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.gen_dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self.snr_range = config.snr_max - config.snr_min
        self.snr_center = (config.snr_max + config.snr_min) / 2

        logger.info(f"SNRGenerator: input_dim={self.input_dim}, arch={config.gen_hidden_dims} -> 1")

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.LeakyReLU(0.2))

    def forward(self, history: torch.Tensor, aux_features: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, aux_features, noise], dim=1)
        raw_output = self.network(x)
        constrained_snr = torch.tanh(raw_output) * (self.snr_range / 2) + self.snr_center
        return constrained_snr


class SNRDiscriminator(nn.Module):
    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        self.input_dim = config.history_len + config.aux_dim + 1

        layers = []
        prev_dim = self.input_dim

        for hidden_dim in config.disc_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(config.disc_activation),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(config.disc_dropout)
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])

        self.network = nn.Sequential(*layers)

        logger.info(f"SNRDiscriminator: input_dim={self.input_dim}, arch={config.disc_hidden_dims} -> 1")

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.LeakyReLU(0.2))

    def forward(self, history: torch.Tensor, aux_features: torch.Tensor,
                next_snr: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, aux_features, next_snr], dim=1)
        probability = self.network(x)
        return probability


class SNRGan:
    def __init__(self, config: GANConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.generator = SNRGenerator(config).to(self.device)
        self.discriminator = SNRDiscriminator(config).to(self.device)

        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2)
        )

        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.training_history = {
            'loss_d': deque(maxlen=1000),
            'loss_g': deque(maxlen=1000),
            'loss_l1': deque(maxlen=1000),
            'iteration': 0,
            'predictions_quality': deque(maxlen=500)
        }

        self.data_buffer = {
            'history': deque(maxlen=10000),
            'aux': deque(maxlen=10000),
            'target': deque(maxlen=10000)
        }

        logger.info(f"SNR-GAN initialized on {self.device}")
        logger.info(f"Generator params: {sum(p.numel() for p in self.generator.parameters())}")
        logger.info(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters())}")

    def _generate_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.config.noise_dim,
                           device=self.device, dtype=self.config.dtype)

    def _validate_inputs(self, history: torch.Tensor, aux: torch.Tensor,
                         y_true: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        if not isinstance(history, torch.Tensor):
            history = torch.tensor(history, dtype=self.config.dtype, device=self.device)
        else:
            history = history.to(device=self.device, dtype=self.config.dtype)

        if not isinstance(aux, torch.Tensor):
            aux = torch.tensor(aux, dtype=self.config.dtype, device=self.device)
        else:
            aux = aux.to(device=self.device, dtype=self.config.dtype)

        if history.dim() == 1:
            history = history.unsqueeze(0)
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)

        batch_size = history.shape[0]

        if history.shape[1] != self.config.history_len:
            raise ValueError(f"History length mismatch: expected {self.config.history_len}, got {history.shape[1]}")
        if aux.shape[1] != self.config.aux_dim:
            raise ValueError(f"Aux dim mismatch: expected {self.config.aux_dim}, got {aux.shape[1]}")
        if aux.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: history={batch_size}, aux={aux.shape[0]}")

        if y_true is not None:
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true, dtype=self.config.dtype, device=self.device)
            else:
                y_true = y_true.to(device=self.device, dtype=self.config.dtype)

            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)
            if y_true.shape[0] != batch_size:
                raise ValueError(f"Target batch size mismatch: expected {batch_size}, got {y_true.shape[0]}")

            return history, aux, y_true

        return history, aux

    def _compute_gradient_penalty(self, history: torch.Tensor, aux: torch.Tensor,
                                   real_snr: torch.Tensor, fake_snr: torch.Tensor) -> torch.Tensor:
        batch_size = real_snr.shape[0]
        alpha = torch.rand(batch_size, 1, device=self.device, dtype=self.config.dtype)
        interpolated = (alpha * real_snr + (1 - alpha) * fake_snr.detach()).requires_grad_(True)
        d_interpolated = self.discriminator(history, aux, interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient_penalty = ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, history: torch.Tensor, aux: torch.Tensor,
                   y_true: torch.Tensor) -> Tuple[float, float]:
        try:
            history, aux, y_true = self._validate_inputs(history, aux, y_true)
            batch_size = history.shape[0]

            if not (torch.isfinite(history).all() and torch.isfinite(aux).all() and torch.isfinite(y_true).all()):
                logger.warning("Non-finite data detected, skipping iteration")
                return 0.0, 0.0

            try:
                self.data_buffer['history'].extend(history.detach().cpu().numpy())
                self.data_buffer['aux'].extend(aux.detach().cpu().numpy())
                self.data_buffer['target'].extend(y_true.detach().cpu().numpy())
            except Exception as e:
                logger.warning(f"Buffer storage error: {e}")

            self.optimizer_d.zero_grad()

            real_labels = torch.ones(batch_size, 1, device=self.device, dtype=self.config.dtype)
            fake_labels = torch.zeros(batch_size, 1, device=self.device, dtype=self.config.dtype)

            real_pred = self.discriminator(history, aux, y_true)
            loss_d_real = self.adversarial_loss(real_pred, real_labels)

            noise = self._generate_noise(batch_size)
            with torch.no_grad():
                fake_snr = self.generator(history, aux, noise)

            fake_pred = self.discriminator(history, aux, fake_snr.detach())
            loss_d_fake = self.adversarial_loss(fake_pred, fake_labels)

            gp = self._compute_gradient_penalty(history, aux, y_true, fake_snr)
            loss_d = (loss_d_real + loss_d_fake) * self.config.adversarial_weight + \
                      self.config.gradient_penalty_weight * gp

            if torch.isfinite(loss_d):
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_d.step()
            else:
                logger.warning("Non-finite discriminator loss, skipping step")
                return 0.0, 0.0

            self.optimizer_g.zero_grad()

            noise = self._generate_noise(batch_size)
            fake_snr = self.generator(history, aux, noise)

            fake_pred = self.discriminator(history, aux, fake_snr)
            loss_g_adversarial = self.adversarial_loss(fake_pred, real_labels)

            loss_l1 = self.l1_loss(fake_snr, y_true)

            loss_g = (loss_g_adversarial * self.config.adversarial_weight +
                      loss_l1 * self.config.l1_penalty_weight)

            if torch.isfinite(loss_g):
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.optimizer_g.step()
            else:
                logger.warning("Non-finite generator loss, skipping step")
                return float(loss_d.item()) if torch.isfinite(loss_d) else 0.0, 0.0

            loss_d_val = float(loss_d.item()) if torch.isfinite(loss_d) else 0.0
            loss_g_val = float(loss_g.item()) if torch.isfinite(loss_g) else 0.0
            loss_l1_val = float(loss_l1.item()) if torch.isfinite(loss_l1) else 0.0

            self.training_history['loss_d'].append(loss_d_val)
            self.training_history['loss_g'].append(loss_g_val)
            self.training_history['loss_l1'].append(loss_l1_val)
            self.training_history['iteration'] += 1

            with torch.no_grad():
                prediction_error = torch.abs(fake_snr - y_true).mean().item()
                if torch.isfinite(torch.tensor(prediction_error)):
                    self.training_history['predictions_quality'].append(float(prediction_error))

            if (self.config.enable_logging and
                    self.training_history['iteration'] % self.config.log_frequency == 0):
                try:
                    avg_loss_d = np.mean(list(self.training_history['loss_d'])[-10:])
                    avg_loss_g = np.mean(list(self.training_history['loss_g'])[-10:])
                    avg_l1 = np.mean(list(self.training_history['loss_l1'])[-10:])
                    avg_pred_error = np.mean(list(self.training_history['predictions_quality'])[-10:])
                    logger.info(f"GAN Iter {self.training_history['iteration']}: "
                                f"D={avg_loss_d:.4f}, G={avg_loss_g:.4f}, L1={avg_l1:.4f}, "
                                f"Pred_MAE={avg_pred_error:.3f}dB")
                except Exception as e:
                    logger.debug(f"Logging error: {e}")

            return loss_d_val, loss_g_val
        except Exception as e:
            logger.error(f"Error in train_step: {e}")
            return 0.0, 0.0

    def predict_next_snr(self, history: torch.Tensor, aux: torch.Tensor,
                         return_confidence: bool = False) -> torch.Tensor:
        self.generator.eval()

        try:
            with torch.no_grad():
                history, aux = self._validate_inputs(history, aux)
                batch_size = history.shape[0]

                predictions = []
                for _ in range(self.config.ensemble_samples):
                    noise = self._generate_noise(batch_size)
                    pred = self.generator(history, aux, noise)
                    if torch.isfinite(pred).all():
                        predictions.append(pred)

                if not predictions:
                    logger.warning("No valid predictions generated")
                    fallback_pred = history.mean(dim=1, keepdim=True)
                    return (fallback_pred, torch.zeros(batch_size, 2)) if return_confidence else fallback_pred

                predictions_tensor = torch.cat(predictions, dim=1)

                if self.config.ensemble_method == "mean":
                    final_prediction = predictions_tensor.mean(dim=1, keepdim=True)
                elif self.config.ensemble_method == "median":
                    final_prediction = predictions_tensor.median(dim=1, keepdim=True)[0]
                else:
                    final_prediction = predictions_tensor.mean(dim=1, keepdim=True)

                if return_confidence:
                    conf_low = torch.quantile(predictions_tensor, 0.1, dim=1, keepdim=True)
                    conf_high = torch.quantile(predictions_tensor, 0.9, dim=1, keepdim=True)
                    confidence_interval = torch.cat([conf_low, conf_high], dim=1)
                    return final_prediction, confidence_interval

                return final_prediction
        except Exception as e:
            logger.error(f"Error in predict_next_snr: {e}")
            try:
                fallback_pred = history.mean(dim=1, keepdim=True)
                return (fallback_pred, torch.zeros(history.shape[0], 2)) if return_confidence else fallback_pred
            except:
                dummy_pred = torch.tensor([[20.0]], device=self.device, dtype=self.config.dtype)
                return (dummy_pred, torch.zeros(1, 2)) if return_confidence else dummy_pred

    def evaluate_prediction_quality(self, history: torch.Tensor, aux: torch.Tensor,
                                    y_true: torch.Tensor) -> Dict[str, float]:
        self.generator.eval()

        try:
            with torch.no_grad():
                history, aux, y_true = self._validate_inputs(history, aux, y_true)

                y_pred, confidence = self.predict_next_snr(history, aux, return_confidence=True)

                if not (torch.isfinite(y_pred).all() and torch.isfinite(y_true).all()):
                    logger.warning("Non-finite data detected during evaluation")
                    return {"status": "evaluation_failed", "error": "non_finite_data"}

                print(f"\n=== EVALUATION DEBUG ===")
                print(f"y_true range: [{y_true.min().item():.3f}, {y_true.max().item():.3f}]")
                print(f"y_true mean±std: {y_true.mean().item():.3f}±{y_true.std().item():.3f}")
                print(f"Low SNR samples: {(y_true < 15.0).sum().item()}")
                print(f"Med SNR samples: {((y_true >= 15.0) & (y_true < 25.0)).sum().item()}")
                print(f"High SNR samples: {(y_true >= 25.0).sum().item()}")
                print(f"y_pred range: [{y_pred.min().item():.3f}, {y_pred.max().item():.3f}]")

                mae = torch.abs(y_pred - y_true).mean().item()
                rmse = torch.sqrt(torch.square(y_pred - y_true).mean()).item()

                try:
                    low_threshold = torch.quantile(y_true, 0.33)
                    high_threshold = torch.quantile(y_true, 0.67)

                    print(f"Adaptive thresholds: low<{low_threshold:.3f}, "
                          f"med=[{low_threshold:.3f},{high_threshold:.3f}], high>{high_threshold:.3f}")

                    low_snr_mask = y_true < low_threshold
                    med_snr_mask = (y_true >= low_threshold) & (y_true < high_threshold)
                    high_snr_mask = y_true >= high_threshold

                    print(f"Sample distribution: low={low_snr_mask.sum().item()}, "
                          f"med={med_snr_mask.sum().item()}, high={high_snr_mask.sum().item()}")

                    def safe_mae(pred, true, mask, range_name="unknown"):
                        if mask.sum() > 0:
                            mae_val = torch.abs(pred[mask] - true[mask]).mean().item()
                            print(f"  {range_name} MAE: {mae_val:.4f} (n={mask.sum().item()})")
                            return mae_val
                        else:
                            print(f"  {range_name} MAE: N/A (no samples)")
                            return float('nan')

                    mae_low = safe_mae(y_pred, y_true, low_snr_mask, "Low SNR")
                    mae_med = safe_mae(y_pred, y_true, med_snr_mask, "Med SNR")
                    mae_high = safe_mae(y_pred, y_true, high_snr_mask, "High SNR")

                    in_conf_interval = ((y_true >= confidence[:, 0:1]) &
                                        (y_true <= confidence[:, 1:2])).float().mean().item()

                    avg_conf_width = (confidence[:, 1] - confidence[:, 0]).mean().item()

                except Exception as e:
                    logger.warning(f"Error computing detailed metrics: {e}")
                    mae_low = mae_med = mae_high = mae
                    in_conf_interval = 0.5
                    avg_conf_width = 5.0

                return {
                    'mae_overall': float(mae),
                    'rmse_overall': float(rmse),
                    'mae_low_snr': float(mae_low),
                    'mae_medium_snr': float(mae_med),
                    'mae_high_snr': float(mae_high),
                    'confidence_coverage': float(in_conf_interval),
                    'avg_confidence_width': float(avg_conf_width),
                    'total_samples': int(history.shape[0]),
                    'data_range': f"[{y_true.min().item():.3f}, {y_true.max().item():.3f}]",
                    'status': 'success'
                }
        except Exception as e:
            logger.error(f"Error in evaluate_prediction_quality: {e}")
            return {
                'status': 'evaluation_failed',
                'error': str(e),
                'mae_overall': float('inf')
            }

    def debug_prediction_space(self, history, aux, y_true):
        pred_norm = self.predict_next_snr(history, aux)
        mae_norm = torch.abs(pred_norm - y_true).mean().item()

        print(f"Normalized space MAE: {mae_norm:.3f}")
        print(f"Normalized pred range: [{pred_norm.min().item():.3f}, {pred_norm.max().item():.3f}]")
        print(f"Normalized true range: [{y_true.min().item():.3f}, {y_true.max().item():.3f}]")

        return mae_norm

    def get_training_summary(self) -> Dict[str, Any]:
        try:
            if not self.training_history['loss_d']:
                return {"status": "no_training_data"}

            recent_window = min(100, len(self.training_history['loss_d']))

            def safe_mean(data, window):
                try:
                    if len(data) == 0:
                        return 0.0
                    recent_data = list(data)[-window:]
                    return float(np.mean(recent_data)) if recent_data else 0.0
                except:
                    return 0.0

            def safe_std(data, window):
                try:
                    if len(data) == 0:
                        return 0.0
                    recent_data = list(data)[-window:]
                    return float(np.std(recent_data)) if len(recent_data) > 1 else 0.0
                except:
                    return 0.0

            return {
                'training_iterations': int(self.training_history['iteration']),
                'data_buffer_size': len(self.data_buffer['history']),
                'recent_performance': {
                    'avg_loss_discriminator': safe_mean(self.training_history['loss_d'], recent_window),
                    'avg_loss_generator': safe_mean(self.training_history['loss_g'], recent_window),
                    'avg_l1_penalty': safe_mean(self.training_history['loss_l1'], recent_window),
                    'avg_prediction_error_db': safe_mean(self.training_history['predictions_quality'], recent_window)
                },
                'model_info': {
                    'generator_params': int(sum(p.numel() for p in self.generator.parameters())),
                    'discriminator_params': int(sum(p.numel() for p in self.discriminator.parameters())),
                    'device': str(self.device),
                    'config_summary': {
                        'history_len': self.config.history_len,
                        'aux_dim': self.config.aux_dim,
                        'ensemble_samples': self.config.ensemble_samples,
                        'snr_range': f"[{self.config.snr_min}, {self.config.snr_max}]"
                    }
                },
                'stability_indicators': {
                    'loss_d_std': safe_std(self.training_history['loss_d'], recent_window),
                    'loss_g_std': safe_std(self.training_history['loss_g'], recent_window),
                    'prediction_consistency': float(1.0 / (1.0 + safe_std(self.training_history['predictions_quality'], recent_window)))
                },
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error in get_training_summary: {e}")
            return {"status": "summary_failed", "error": str(e)}

    def save(self, filepath: str = "snr_gan_model.pth"):
        try:
            def safe_list_conversion(deque_data, max_items=1000):
                try:
                    return list(deque_data)[-max_items:]
                except:
                    return []

            checkpoint = {
                'config': {k: (str(v) if isinstance(v, torch.dtype) else v)
                           for k, v in self.config.__dict__.items()},
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                'training_history': {
                    'loss_d': safe_list_conversion(self.training_history['loss_d'], 1000),
                    'loss_g': safe_list_conversion(self.training_history['loss_g'], 1000),
                    'loss_l1': safe_list_conversion(self.training_history['loss_l1'], 1000),
                    'iteration': self.training_history['iteration'],
                    'predictions_quality': safe_list_conversion(self.training_history['predictions_quality'], 500)
                },
                'data_buffer': {
                    'history': safe_list_conversion(self.data_buffer['history'], 1000),
                    'aux': safe_list_conversion(self.data_buffer['aux'], 1000),
                    'target': safe_list_conversion(self.data_buffer['target'], 1000)
                },
                'data_scaler_snr_mean': self.scaler_snr.mean_ if hasattr(self, 'scaler_snr') else None,
                'data_scaler_snr_scale': self.scaler_snr.scale_ if hasattr(self, 'scaler_snr') else None,
            }

            torch.save(checkpoint, filepath)
            logger.info(f"SNR-GAN model saved: {filepath}")

            config_path = filepath.replace('.pth', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(checkpoint['config'], f, indent=2, default=str)

            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, filepath: str = "enhanced_snr_gan.pth") -> bool:
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

            if 'training_history' in checkpoint:
                hist = checkpoint['training_history']
                self.training_history['loss_d'] = deque(hist.get('loss_d', []), maxlen=1000)
                self.training_history['loss_g'] = deque(hist.get('loss_g', []), maxlen=1000)
                self.training_history['loss_l1'] = deque(hist.get('loss_l1', []), maxlen=1000)
                self.training_history['iteration'] = hist.get('iteration', 0)
                self.training_history['predictions_quality'] = deque(
                    hist.get('predictions_quality', []), maxlen=500
                )

            if 'data_buffer' in checkpoint:
                buff = checkpoint['data_buffer']
                self.data_buffer['history'] = deque(buff.get('history', []), maxlen=10000)
                self.data_buffer['aux'] = deque(buff.get('aux', []), maxlen=10000)
                self.data_buffer['target'] = deque(buff.get('target', []), maxlen=10000)

            logger.info(f"SNR-GAN model loaded: {filepath}")
            logger.info(f"Training iterations: {self.training_history['iteration']}")
            logger.info(f"Data buffer size: {len(self.data_buffer['history'])}")

            return True
        except FileNotFoundError:
            logger.warning(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def continual_learning_update(self, recent_data_points: int = 100):
        try:
            if len(self.data_buffer['history']) < recent_data_points:
                logger.warning("Not enough data for continual learning update")
                return False

            recent_history = torch.tensor(
                list(self.data_buffer['history'])[-recent_data_points:],
                dtype=self.config.dtype, device=self.device
            )
            recent_aux = torch.tensor(
                list(self.data_buffer['aux'])[-recent_data_points:],
                dtype=self.config.dtype, device=self.device
            )
            recent_target = torch.tensor(
                list(self.data_buffer['target'])[-recent_data_points:],
                dtype=self.config.dtype, device=self.device
            )

            if not (torch.isfinite(recent_history).all() and
                    torch.isfinite(recent_aux).all() and
                    torch.isfinite(recent_target).all()):
                logger.warning("Non-finite data detected, cancelling update")
                return False

            original_lr_g = self.optimizer_g.param_groups[0]['lr']
            original_lr_d = self.optimizer_d.param_groups[0]['lr']

            self.optimizer_g.param_groups[0]['lr'] = original_lr_g * 0.1
            self.optimizer_d.param_groups[0]['lr'] = original_lr_d * 0.1

            for i in range(10):
                try:
                    self.train_step(recent_history, recent_aux, recent_target)
                except Exception as e:
                    logger.warning(f"Error in continual learning iteration {i}: {e}")
                    break

            self.optimizer_g.param_groups[0]['lr'] = original_lr_g
            self.optimizer_d.param_groups[0]['lr'] = original_lr_d

            logger.info("Continual learning update complete")
            return True
        except Exception as e:
            logger.error(f"Error in continual_learning_update: {e}")
            return False


class SNRDataProcessor:
    def __init__(self, history_len: int = 10, aux_features: List[str] = None):
        self.history_len = history_len
        self.aux_features = aux_features or ['mobility_speed', 'distance', 'channel_stability']

        self.flow_histories = {}
        self.flow_aux_data = {}

        self.snr_stats = {'mean': 20.0, 'std': 8.0}
        self.aux_stats = {}

        logger.info(f"SNRDataProcessor: history_len={history_len}, features={self.aux_features}")

    def update_flow_data(self, flow_id: int, snr: float,
                         mobility_speed: float = 3.0, distance: float = 100.0,
                         channel_stability: float = 0.5, **kwargs):
        try:
            if not (np.isfinite(snr) and np.isfinite(mobility_speed) and
                    np.isfinite(distance) and np.isfinite(channel_stability)):
                logger.warning(f"Non-finite data for flow {flow_id}, ignored")
                return False

            if flow_id not in self.flow_histories:
                self.flow_histories[flow_id] = deque(maxlen=self.history_len * 2)
                self.flow_aux_data[flow_id] = deque(maxlen=self.history_len * 2)

            snr_clipped = np.clip(float(snr), -15.0, 40.0)
            self.flow_histories[flow_id].append(snr_clipped)

            aux_data = {
                'mobility_speed': np.clip(float(mobility_speed), 0.0, 200.0),
                'distance': np.clip(float(distance), 10.0, 10000.0),
                'channel_stability': np.clip(float(channel_stability), 0.0, 1.0)
            }
            aux_data.update({k: float(v) for k, v in kwargs.items() if np.isfinite(v)})

            self.flow_aux_data[flow_id].append(aux_data)
            return True
        except Exception as e:
            logger.warning(f"Error in update_flow_data for flow {flow_id}: {e}")
            return False

    def can_predict(self, flow_id: int) -> bool:
        try:
            return (flow_id in self.flow_histories and
                    len(self.flow_histories[flow_id]) >= self.history_len and
                    flow_id in self.flow_aux_data and
                    len(self.flow_aux_data[flow_id]) > 0)
        except:
            return False

    def prepare_prediction_data(self, flow_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.can_predict(flow_id):
            raise ValueError(f"Insufficient history for flow {flow_id}")

        try:
            history_list = list(self.flow_histories[flow_id])[-self.history_len:]
            if len(history_list) != self.history_len:
                raise ValueError(f"Incomplete history: {len(history_list)} vs {self.history_len}")

            if not all(np.isfinite(x) for x in history_list):
                raise ValueError("Non-finite SNR data detected")

            history_tensor = torch.tensor(history_list, dtype=torch.float32)

            latest_aux = self.flow_aux_data[flow_id][-1]
            aux_values = []

            for feat in self.aux_features:
                value = latest_aux.get(feat, 0.0)
                if not np.isfinite(value):
                    logger.warning(f"Non-finite feature {feat}, using default")
                    value = {'mobility_speed': 3.0, 'distance': 200.0, 'channel_stability': 0.5}.get(feat, 0.0)
                aux_values.append(float(value))

            aux_tensor = torch.tensor(aux_values, dtype=torch.float32)

            return history_tensor, aux_tensor
        except Exception as e:
            logger.error(f"Error in prepare_prediction_data for flow {flow_id}: {e}")
            raise


def create_synthetic_snr_data(num_samples: int = 1000, history_len: int = 10,
                               snr_base: float = 20.0, noise_level: float = 3.0,
                               trend_strength: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        snr_series = [snr_base]

        for i in range(num_samples + history_len):
            trend = trend_strength * np.sin(i * 0.01) * 2.0
            noise = noise_level * np.random.normal() * 0.7 + snr_series[-1] * 0.1 - snr_base * 0.1
            next_snr = np.clip(snr_series[-1] + trend + noise, -5.0, 35.0)
            snr_series.append(next_snr)

        snr_series = np.array(snr_series[1:])

        mobility_speed = 3.0 + 20.0 * np.random.random(num_samples + history_len)
        distance = 50.0 + 200.0 * np.random.random(num_samples + history_len)
        stability = np.clip(1.0 - np.abs(np.diff(snr_series, prepend=snr_series[0])) / 5.0, 0.0, 1.0)

        aux_features = np.column_stack([mobility_speed, distance, stability])

        X_history = []
        X_aux = []
        y_target = []

        for i in range(num_samples):
            hist_seq = snr_series[i:i + history_len]
            target_snr = snr_series[i + history_len]
            aux_vals = aux_features[i + history_len - 1]

            if (np.isfinite(hist_seq).all() and
                    np.isfinite(target_snr) and
                    np.isfinite(aux_vals).all()):
                X_history.append(hist_seq)
                X_aux.append(aux_vals)
                y_target.append(target_snr)

        return np.array(X_history), np.array(X_aux), np.array(y_target)
    except Exception as e:
        logger.error(f"Error in create_synthetic_snr_data: {e}")
        X_history = np.random.normal(20.0, 3.0, (100, history_len))
        X_aux = np.random.random((100, 3)) * [50, 500, 1] + [3, 50, 0]
        y_target = np.random.normal(20.0, 3.0, 100)
        return X_history, X_aux, y_target


def example_usage():
    config = GANConfig(
        history_len=10,
        aux_dim=3,
        noise_dim=16,
        gen_hidden_dims=[128, 256, 128],
        disc_hidden_dims=[128, 256, 128, 64],
        learning_rate_g=0.0002,
        learning_rate_d=0.0002,
        ensemble_samples=20,
        ensemble_method="median",
        dtype=torch.float32
    )

    gan = SNRGan(config)
    processor = SNRDataProcessor(history_len=config.history_len)

    print("Generating synthetic data...")
    X_history, X_aux, y_target = create_synthetic_snr_data(
        num_samples=1000,
        history_len=config.history_len
    )
    print(f"Data generated: {X_history.shape}, {X_aux.shape}, {y_target.shape}")

    print("Training GAN...")
    X_history_tensor = torch.tensor(X_history, dtype=torch.float32)
    X_aux_tensor = torch.tensor(X_aux, dtype=torch.float32)
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32).unsqueeze(-1)

    successful_epochs = 0
    for epoch in range(50):
        try:
            batch_size = min(32, len(X_history))
            indices = np.random.choice(len(X_history), batch_size, replace=False)

            batch_history = X_history_tensor[indices]
            batch_aux = X_aux_tensor[indices]
            batch_target = y_target_tensor[indices]

            loss_d, loss_g = gan.train_step(batch_history, batch_aux, batch_target)

            if loss_d > 0 and loss_g > 0:
                successful_epochs += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss_D={loss_d:.4f}, Loss_G={loss_g:.4f}")
        except Exception as e:
            logger.warning(f"Error at epoch {epoch}: {e}")
            continue

    print(f"Training done: {successful_epochs}/50 successful epochs")

    test_history = X_history_tensor[:5]
    test_aux = X_aux_tensor[:5]
    test_target = y_target_tensor[:5]

    try:
        predictions = gan.predict_next_snr(test_history, test_aux)
        print("Predictions vs ground truth:")
        for i in range(5):
            pred = predictions[i].item()
            real = test_target[i].item()
            error = abs(pred - real)
            print(f"  Sample {i}: Pred={pred:.2f}dB, Real={real:.2f}dB, Error={error:.2f}dB")
    except Exception as e:
        print(f"Prediction error: {e}")

    try:
        metrics = gan.evaluate_prediction_quality(test_history, test_aux, test_target)
        if metrics.get('status') == 'success':
            print(f"\nPerformance metrics:")
            print(f"  MAE: {metrics['mae_overall']:.3f} dB")
            print(f"  RMSE: {metrics['rmse_overall']:.3f} dB")
        else:
            print(f"Evaluation failed: {metrics.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Evaluation error: {e}")

    try:
        success = gan.save("example_snr_gan.pth")
        if success:
            print("Model saved successfully")
        else:
            print("Error saving model")
    except Exception as e:
        print(f"Save error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )
    example_usage()
