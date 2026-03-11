#!/usr/bin/env python3

import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import sionna.phy
    from sionna.phy.channel import RayleighBlockFading, AWGN
    from sionna.phy import *
    # from sionna.rt import *
    SIONNA_AVAILABLE = True
    print("Sionna PHY imported successfully")
except ImportError as e:
    print(f"Sionna PHY import error: {e}")
    SIONNA_AVAILABLE = False

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tf.get_logger().setLevel('ERROR')

import numpy as np
import socket
import json
import threading
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sionna_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RealisticPHYProcessor:
    def __init__(self):
        self.sionna_available = SIONNA_AVAILABLE
        self.default_params = {
            'carrier_frequency': 2.4e9,
            'speed': 3.0,
            'distance': 100.0,
            'num_paths': 6,
            'delay_spread': 1e-6,
            'k_factor': 10.0,
            'phase_noise_std': 0.01,
        }

        if self.sionna_available:
            self.rayleigh_channel = RayleighBlockFading(
                num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1
            )
            self.awgn_channel = AWGN()

        logger.info(f"PHY Processor initialized (Sionna: {'available' if self.sionna_available else 'unavailable'})")

    def add_realistic_impairments(self, symbols, snr_db, channel_params=None):
        if channel_params is None:
            channel_params = self.default_params

        phase_noise_std = channel_params['phase_noise_std'] * 0.1
        phase_noise = tf.random.normal(tf.shape(symbols), stddev=phase_noise_std)
        phase_noise_complex = tf.complex(tf.zeros_like(phase_noise), phase_noise)
        symbols_with_phase_noise = symbols * tf.exp(phase_noise_complex)

        time_samples = tf.range(tf.shape(symbols)[1], dtype=tf.float32)
        cfo_hz = tf.random.uniform([], -10.0, 10.0)
        cfo_phase = 2.0 * np.pi * cfo_hz * time_samples / channel_params['carrier_frequency']
        cfo_rotation = tf.exp(tf.complex(tf.zeros_like(cfo_phase), cfo_phase))
        symbols_with_cfo = symbols_with_phase_noise * tf.expand_dims(cfo_rotation, 0)

        return symbols_with_cfo

    def simulate_realistic_channel(self, symbols, snr_db, channel_type="rayleigh", channel_params=None):
        if channel_params is None:
            channel_params = self.default_params

        try:
            if channel_type.lower() == "rayleigh":
                h, tau = self.rayleigh_channel(
                    batch_size=tf.shape(symbols)[0],
                    num_time_steps=tf.shape(symbols)[1]
                )
                h_squeezed = tf.squeeze(h, axis=[1, 2, 3, 4, 5])
                symbols_faded = symbols * h_squeezed
                avg_channel_gain = tf.reduce_mean(tf.abs(h_squeezed))
                logger.debug(f"Rayleigh channel: avg gain = {avg_channel_gain:.3f}")

            elif channel_type.lower() == "awgn":
                symbols_faded = symbols

            elif channel_type.lower() in ["rice", "rician"]:
                k_factor_linear = 10 ** (channel_params['k_factor'] / 10)
                los_component = tf.complex(tf.sqrt(k_factor_linear / (k_factor_linear + 1)), 0.0)
                h_rayleigh, _ = self.rayleigh_channel(
                    batch_size=tf.shape(symbols)[0],
                    num_time_steps=tf.shape(symbols)[1]
                )
                h_rayleigh_squeezed = tf.squeeze(h_rayleigh, axis=[1, 2, 3, 4, 5])
                nlos_component = h_rayleigh_squeezed * tf.sqrt(1 / (k_factor_linear + 1))
                h_rice = los_component + nlos_component
                symbols_faded = symbols * h_rice

            elif channel_type.lower() == "urban":
                h, tau = self.rayleigh_channel(
                    batch_size=tf.shape(symbols)[0],
                    num_time_steps=tf.shape(symbols)[1]
                )
                h_squeezed = tf.squeeze(h, axis=[1, 2, 3, 4, 5])
                urban_attenuation = tf.complex(0.6, 0.0)
                h_urban = h_squeezed * urban_attenuation
                symbols_faded = symbols * h_urban

            else:
                symbols_faded = symbols
                logger.warning(f"Unknown channel type: {channel_type}, using AWGN")

            symbols_impaired = self.add_realistic_impairments(symbols_faded, snr_db, channel_params)

            modulation_order = self.get_modulation_order(snr_db)
            num_bits_per_symbol = int(np.log2(modulation_order))

            no = sionna.phy.utils.ebnodb2no(snr_db,
                                            num_bits_per_symbol=num_bits_per_symbol,
                                            coderate=1.0)

            symbols_noisy = self.awgn_channel(symbols_impaired, no)

            return symbols_noisy, no, h_squeezed if 'h_squeezed' in locals() else None

        except Exception as e:
            logger.error(f"Channel simulation error: {e}")
            num_bits_per_symbol = 2
            no = sionna.phy.utils.ebnodb2no(snr_db, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)
            return self.awgn_channel(symbols, no), no, None

    def get_modulation_order(self, snr_db):
        if snr_db < 10:
            return 4
        elif snr_db < 18:
            return 16
        elif snr_db < 25:
            return 64
        else:
            return 256

    def calculate_realistic_bler(self, bits_original, bits_received, packet_size=128):
        total_bits = len(bits_original)
        if total_bits < packet_size:
            packet_size = total_bits

        num_packets = max(1, total_bits // packet_size)
        packets_in_error = 0

        for i in range(num_packets):
            start_idx = i * packet_size
            end_idx = min((i + 1) * packet_size, total_bits)

            packet_original = bits_original[start_idx:end_idx]
            packet_received = bits_received[start_idx:end_idx]

            packet_errors = tf.reduce_sum(tf.cast(tf.not_equal(packet_original, packet_received), tf.int32))
            if packet_errors > 0:
                packets_in_error += 1

        bler = packets_in_error / num_packets
        return bler, packets_in_error, num_packets

    def calculate_realistic_throughput(self, modulation: str, ber: float, bler: float,
                                       snr: float, success: bool, num_bits: int = None) -> float:
        modulation_rates = {
            "qam4": 2.0,
            "qam16": 4.0,
            "qam64": 6.0,
            "qam256": 8.0
        }

        base_rate = modulation_rates.get(modulation, 4.0)

        if not success:
            return 0.0

        if ber < 1e-6:
            quality_factor = 0.98
        elif ber < 1e-4:
            quality_factor = 0.95
        elif ber < 1e-3:
            quality_factor = 0.90
        else:
            quality_factor = 0.80

        snr_efficiency = min(1.0, (snr + 10) / 40.0)
        channel_overhead = 0.85

        effective_throughput = (
            base_rate *
            (1.0 - bler) *
            quality_factor *
            snr_efficiency *
            channel_overhead
        )

        variability = 0.975 + 0.05 * np.random.random()
        effective_throughput *= variability

        return max(0.0, effective_throughput)

    def process_transmission_realistic(self, bits: np.ndarray, modulation: str, snr_db: float,
                                       channel_type: str = "rayleigh", channel_params: dict = None):
        try:
            k = len(bits)
            min_bits = 1024

            if k < min_bits:
                repetitions = (min_bits + k - 1) // k
                bits_extended = np.tile(bits, repetitions)[:min_bits]
                k_processing = min_bits
                logger.info(f"Bits extended: {k} -> {k_processing} for reliable statistics")
            else:
                bits_extended = bits
                k_processing = k

            bits_tf = tf.cast(bits_extended, tf.float32)

            modulation_order = int(modulation.replace("qam", "")) if "qam" in modulation else 4
            num_bits_per_symbol = int(np.log2(modulation_order))

            if k_processing % num_bits_per_symbol != 0:
                padding_needed = num_bits_per_symbol - (k_processing % num_bits_per_symbol)
                bits_padded = tf.concat([bits_tf, tf.zeros(padding_needed)], axis=0)
                k_padded = k_processing + padding_needed
            else:
                bits_padded = bits_tf
                k_padded = k_processing

            bits_reshaped = tf.reshape(bits_padded, [1, k_padded])

            if modulation_order in [4, 16, 64, 256]:
                constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol)
            else:
                constellation = sionna.phy.mapping.Constellation("qam", 2)
                num_bits_per_symbol = 2
                logger.warning(f"Unknown modulation {modulation}, using QPSK")

            mapper = sionna.phy.mapping.Mapper(constellation=constellation)
            symbols = mapper(bits_reshaped)

            noisy_symbols, no, channel_response = self.simulate_realistic_channel(
                symbols, snr_db, channel_type, channel_params
            )

            if channel_type.lower() in ["rayleigh", "urban", "rice", "rician"] and channel_response is not None:
                epsilon = 1e-8
                channel_conj = tf.math.conj(channel_response)
                channel_power = tf.abs(channel_response) ** 2 + epsilon
                equalized_symbols = noisy_symbols * channel_conj / tf.complex(channel_power, tf.zeros_like(channel_power))
            else:
                equalized_symbols = noisy_symbols

            demapper = sionna.phy.mapping.Demapper("maxlog", constellation=constellation)
            llrs = demapper(equalized_symbols, no)
            bits_hat = tf.cast(llrs > 0, tf.float32)

            bits_hat_original = bits_hat[0, :k_processing]
            bits_original = bits_reshaped[0, :k_processing]

            bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(bits_original, bits_hat_original), tf.int32))
            ber = float(bit_errors.numpy()) / k_processing

            bler, packets_in_error, total_packets = self.calculate_realistic_bler(
                bits_original, bits_hat_original, packet_size=128
            )

            effective_throughput = self.calculate_realistic_throughput(
                modulation, float(ber), float(bler), snr_db,
                bler < 1.0, k_processing
            )

            result = {
                'success': bler < 1.0,
                'bit_errors': int(bit_errors.numpy()),
                'total_bits': k_processing,
                'original_bits': k,
                'ber': ber,
                'bler': bler,
                'packets_in_error': packets_in_error,
                'total_packets': total_packets,
                'snr_db': snr_db,
                'modulation': modulation,
                'channel_type': channel_type,
                'effective_throughput': effective_throughput,
                'simulated': False,
                'realistic': True,
                'extended_for_stats': k_processing > k
            }

            logger.info(f"Sionna result: BER={ber:.6f}, BLER={bler:.3f} "
                        f"({packets_in_error}/{total_packets} packets), T={effective_throughput:.1f}Mbps")

            return result

        except Exception as e:
            logger.error(f"Realistic processing error: {e}")
            return self.simulate_transmission_simple(bits, modulation, snr_db, channel_type)

    def simulate_transmission_simple(self, bits: np.ndarray, modulation: str, snr_db: float, channel_type: str = "awgn"):
        k = len(bits)

        snr_linear = 10 ** (snr_db / 10)
        modulation_order = int(modulation.replace("qam", "")) if "qam" in modulation else 4

        if channel_type.lower() == "awgn":
            if modulation_order == 4:
                ber_theory = 0.5 * np.exp(-snr_linear)
            elif modulation_order == 16:
                ber_theory = 0.75 * np.exp(-0.8 * snr_linear)
            elif modulation_order == 64:
                ber_theory = 0.85 * np.exp(-0.6 * snr_linear)
            else:
                ber_theory = 0.9 * np.exp(-0.4 * snr_linear)

        elif channel_type.lower() in ["rayleigh", "urban"]:
            if modulation_order == 4:
                ber_theory = 0.5 / (1 + snr_linear)
            else:
                ber_theory = 0.8 / (1 + 0.8 * snr_linear / (modulation_order - 1))

        else:
            ber_awgn = 0.5 * np.exp(-0.8 * snr_linear / modulation_order)
            ber_rayleigh = 0.5 / (1 + 0.8 * snr_linear / modulation_order)
            ber_theory = 0.7 * ber_awgn + 0.3 * ber_rayleigh

        ber_theory = min(max(ber_theory, 1e-8), 0.5)
        ber_actual = ber_theory * (0.8 + 0.4 * np.random.random())

        num_errors = int(k * ber_actual)
        if k > 100 and np.random.random() > 0.5:
            num_errors += np.random.poisson(np.sqrt(k * ber_actual))

        num_errors = min(max(num_errors, 0), k)
        ber = num_errors / k if k > 0 else 0

        packet_size = min(64, k)
        num_packets = max(1, k // packet_size)
        packets_in_error = min(num_packets, max(1, int(num_packets * ber * 8)))
        bler = packets_in_error / num_packets

        effective_throughput = (k - num_errors) * 0.1

        return {
            'success': num_errors < k,
            'bit_errors': num_errors,
            'total_bits': k,
            'original_bits': k,
            'ber': ber,
            'bler': bler,
            'packets_in_error': packets_in_error,
            'total_packets': num_packets,
            'snr_db': snr_db,
            'modulation': modulation,
            'channel_type': f"{channel_type}_simulated",
            'effective_throughput': effective_throughput,
            'simulated': True,
            'realistic': False,
            'extended_for_stats': False
        }

    def process_transmission(self, bits: np.ndarray, modulation: str, snr_db: float,
                             channel_type: str = "rayleigh", channel_params: dict = None):
        start_time = time.time()

        if self.sionna_available and len(bits) >= 8:
            result = self.process_transmission_realistic(bits, modulation, snr_db, channel_type, channel_params)
        else:
            result = self.simulate_transmission_simple(bits, modulation, snr_db, channel_type)

        processing_time = time.time() - start_time
        result['processing_time_ms'] = processing_time * 1000

        return result


class RealisticSionnaServer:
    def __init__(self, host='127.0.0.1', port=9000):
        self.host = host
        self.port = port
        self.phy_processor = RealisticPHYProcessor()
        self.socket = None
        self.running = False
        self.stats = {
            'total_requests': 0,
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'avg_ber': 0.0,
            'avg_bler': 0.0
        }

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True

            logger.info(f"Sionna server listening on {self.host}:{self.port}")

            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.error as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.stop()

    def handle_client(self, conn, addr):
        try:
            data = conn.recv(8192).decode()
            if not data:
                conn.close()
                return

            try:
                msg = json.loads(data)
                self.stats['total_requests'] += 1

                channel_type = msg.get('channel_type', 'rayleigh')
                logger.info(f"NS-3 -> Sionna: ID={msg['id']} | channel={channel_type} | "
                            f"mod={msg['modulation']} | SNR={msg['snr_db']}dB | bits={msg['k']}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                error_msg = {"error": "Invalid JSON"}
                conn.sendall(json.dumps(error_msg).encode())
                conn.close()
                return

            required_keys = ['id', 'modulation', 'snr_db', 'k', 'payload']
            for key in required_keys:
                if key not in msg:
                    raise KeyError(f"Missing key: {key}")

            packet_id = msg['id']
            k = msg['k']
            modulation = msg['modulation']
            snr_db = msg['snr_db']
            payload = msg['payload']
            channel_type = msg.get('channel_type', 'rayleigh')
            channel_params = msg.get('channel_params', None)

            bits = np.array(payload, dtype=np.float32)

            result = self.phy_processor.process_transmission(
                bits, modulation, snr_db, channel_type, channel_params
            )

            response = {
                "id": packet_id,
                "success": result['success'],
                "bler": round(result['bler'], 6),
                "ber": round(result['ber'], 8),
                "errors": result['bit_errors'],
                "total_bits": result['total_bits'],
                "original_bits": result.get('original_bits', k),
                "packets_in_error": result.get('packets_in_error', 0),
                "total_packets": result.get('total_packets', 1),
                "snr_db": snr_db,
                "modulation": result['modulation'],
                "channel_type": result['channel_type'],
                "effective_throughput": round(result.get('effective_throughput', 0.0), 2),
                "realistic": result.get('realistic', False),
                "processing_time_ms": round(result.get('processing_time_ms', 0.0), 2),
                "extended_for_stats": result.get('extended_for_stats', False)
            }

            self.update_stats(result)

            response_str = json.dumps(response)
            conn.sendall(response_str.encode())

            mode = "Sionna" if result.get('realistic', False) else "Simulated"
            logger.info(f"Sionna -> NS-3: {mode} | BER={response['ber']:.6f} | "
                        f"BLER={response['bler']:.3f} | T={response['effective_throughput']:.1f}Mbps | "
                        f"{response['processing_time_ms']:.1f}ms")

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            error_msg = {
                "error": str(e),
                "id": msg.get('id', -1) if 'msg' in locals() else -1,
                "success": False,
                "ber": 0.5,
                "bler": 1.0
            }
            try:
                conn.sendall(json.dumps(error_msg).encode())
            except:
                pass
        finally:
            conn.close()

    def update_stats(self, result):
        if result['success']:
            self.stats['successful_transmissions'] += 1
        else:
            self.stats['failed_transmissions'] += 1

        total = self.stats['total_requests']
        self.stats['avg_ber'] = (self.stats['avg_ber'] * (total - 1) + result['ber']) / total
        self.stats['avg_bler'] = (self.stats['avg_bler'] * (total - 1) + result['bler']) / total

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()

        logger.info(f"Total requests: {self.stats['total_requests']}")
        logger.info(f"Successful transmissions: {self.stats['successful_transmissions']}")
        logger.info(f"Failed transmissions: {self.stats['failed_transmissions']}")
        logger.info(f"Avg BER: {self.stats['avg_ber']:.6f}")
        logger.info(f"Avg BLER: {self.stats['avg_bler']:.3f}")
        logger.info("Sionna server stopped")


def main():
    print(f"TensorFlow: {tf.__version__}")
    print(f"Sionna: {'available' if SIONNA_AVAILABLE else 'unavailable (simulation mode)'}")

    server = RealisticSionnaServer()

    try:
        server.start()
    except KeyboardInterrupt:
        print("Server stopped.")
        server.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        server.stop()


if __name__ == "__main__":
    main()
