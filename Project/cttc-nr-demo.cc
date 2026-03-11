// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/simulator.h"
#include "ns3/point-to-point-module.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("CttcNrDemoDynamicEnhanced");

struct MobilityChannelData {
    uint32_t flowId;
    double timestamp;
    Vector position;
    Vector velocity;
    double distance_to_bs;
    double snr_db;
    double pathloss_db;
    std::string channel_condition;
    std::string environment_type;
    double mobility_speed_kmh;
    double channel_stability;
    double ber;
    double bler;
    std::string modulation;
    double throughput_mbps;
    bool transmission_success;
};

struct RealisticChannelConfig {
    std::string channel_type = "urban";
    double carrier_frequency = 28e9;
    double speed = 3.0;
    double distance = 100.0;
    int num_paths = 6;
    double delay_spread = 1e-6;
    double k_factor = 10.0;
    double phase_noise_std = 0.01;
    double shadowing_std = 8.0;
    double fast_fading_std = 2.0;
    double correlation_distance = 50.0;
    double coherence_time = 0.1;
    bool enable_temporal_correlation = true;
};

enum ChannelEnvironment {
    INDOOR,
    URBAN,
    SUBURBAN,
    RURAL
};

enum MobilityType {
    STATIC,
    LINEAR,
    CIRCULAR,
    RANDOM_WALK,
    HIGHWAY,
    URBAN_GRID
};

struct DynamicMobilityConfig {
    MobilityType type = URBAN_GRID;
    double max_speed_kmh = 50.0;
    double min_speed_kmh = 5.0;
    double direction_change_interval = 10.0;
    double area_width = 1000.0;
    double area_height = 1000.0;
    bool enable_acceleration = true;
    double acceleration_factor = 0.1;
};

struct RealisticFlowStats {
    uint32_t txPackets = 0;
    uint64_t txBytes = 0;
    uint32_t rxPackets = 0;
    uint64_t rxBytes = 0;
    double totalDelay = 0.0;
    double totalJitter = 0.0;
    double lastPacketDelay = 0.0;
    uint32_t packetsForDelay = 0;
    std::string lastChannelType = "";
    double avgBER = 0.0;
    double avgBLER = 0.0;
    double lastThroughput = 0.0;
    std::string lastModulation = "";
};

struct FlowHistory {
    double lastThroughput = 0.0;
    double lastBER = 0.0;
    double lastBLER = 0.0;
    std::string lastModulation = "";
    uint32_t packetCount = 0;
    RealisticChannelConfig channelConfig;
    ChannelEnvironment environment = URBAN;
};

std::vector<MobilityChannelData> mobility_data_log;
std::map<uint32_t, std::deque<double>> snr_history;
std::map<uint32_t, Vector> previous_positions;
std::map<uint32_t, Time> last_update_time;
std::map<uint32_t, RealisticFlowStats> realFlowStats;
std::map<uint32_t, FlowHistory> flowHistories;

RealisticChannelConfig global_channel_config;
DynamicMobilityConfig global_mobility_config;
double simTime = 60.0;

RealisticChannelConfig GetChannelConfigForEnvironment(ChannelEnvironment env, double distance) {
    RealisticChannelConfig config = global_channel_config;
    config.distance = distance;

    switch(env) {
        case INDOOR:
            config.channel_type = "rayleigh";
            config.k_factor = 15.0;
            config.num_paths = 4;
            config.speed = 1.0;
            config.phase_noise_std = 0.005;
            config.shadowing_std = 6.0;
            break;

        case URBAN:
            config.channel_type = "urban";
            config.k_factor = 5.0;
            config.num_paths = 12;
            config.speed = 30.0;
            config.phase_noise_std = 0.015;
            config.shadowing_std = 8.0;
            break;

        case SUBURBAN:
            config.channel_type = "rayleigh";
            config.k_factor = 8.0;
            config.num_paths = 6;
            config.speed = 50.0;
            config.phase_noise_std = 0.01;
            config.shadowing_std = 6.0;
            break;

        case RURAL:
            config.channel_type = "awgn";
            config.k_factor = 12.0;
            config.num_paths = 3;
            config.speed = 70.0;
            config.phase_noise_std = 0.008;
            config.shadowing_std = 4.0;
            break;
    }

    return config;
}

ChannelEnvironment DetermineEnvironment(Vector position, double area_width, double area_height) {
    double center_distance = sqrt(pow(position.x - area_width/2, 2) + pow(position.y - area_height/2, 2));
    double normalized_distance = center_distance / (sqrt(area_width*area_width + area_height*area_height) / 2);

    if (normalized_distance < 0.2) {
        return INDOOR;
    } else if (normalized_distance < 0.5) {
        return URBAN;
    } else if (normalized_distance < 0.8) {
        return SUBURBAN;
    } else {
        return RURAL;
    }
}

double CalculateRealisticSNR(Vector ue_position, Vector bs_position, double timestamp,
                             uint32_t flow_id, const RealisticChannelConfig& config) {

    double distance = CalculateDistance(ue_position, bs_position);
    distance = std::max(distance, 1.0);

    double path_loss_db = 20 * log10(config.carrier_frequency / 1e9) +
                         20 * log10(distance) +
                         32.45;

    if (config.channel_type == "urban") {
        path_loss_db += 20.0;
    } else if (config.channel_type == "indoor") {
        path_loss_db += 15.0;
    }

    double shadowing_db = 0.0;
    if (config.enable_temporal_correlation) {
        double spatial_factor = sin(ue_position.x / config.correlation_distance) *
                               cos(ue_position.y / config.correlation_distance);
        double temporal_factor = sin(timestamp / config.coherence_time);
        shadowing_db = config.shadowing_std * spatial_factor * 0.7 +
                      config.shadowing_std * temporal_factor * 0.3;
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, config.shadowing_std);
        shadowing_db = dis(gen);
    }

    double fast_fading_db = 0.0;
    bool is_los = (distance < 200.0 && config.channel_type != "indoor");

    std::random_device rd;
    std::mt19937 gen(rd());
    if (is_los) {
        std::normal_distribution<> dis(0, config.fast_fading_std / 2.0);
        fast_fading_db = dis(gen);
    } else {
        std::normal_distribution<> dis(0, config.fast_fading_std);
        fast_fading_db = dis(gen);
    }

    double tx_power_dbm = 43.0;
    double noise_power_dbm = -104.0;
    double received_power_dbm = tx_power_dbm - path_loss_db - shadowing_db - fast_fading_db;
    double snr_db = received_power_dbm - noise_power_dbm;

    return std::clamp(snr_db, -10.0, 40.0);
}

double CalculateChannelStability(uint32_t flow_id) {
    if (snr_history[flow_id].size() < 3) {
        return 0.5;
    }

    std::vector<double> recent_snr(snr_history[flow_id].end() - std::min(size_t(5), snr_history[flow_id].size()),
                                  snr_history[flow_id].end());

    double mean = std::accumulate(recent_snr.begin(), recent_snr.end(), 0.0) / recent_snr.size();
    double variance = 0.0;
    for (double snr : recent_snr) {
        variance += std::pow(snr - mean, 2);
    }
    variance /= recent_snr.size();

    double stability = 1.0 / (1.0 + variance / 25.0);
    return std::clamp(stability, 0.0, 1.0);
}

std::string RequestModulationFromAMC(double snr, uint32_t flowId,
                                     double previousThroughput,
                                     double previousBER,
                                     double previousBLER,
                                     std::string previousModulation,
                                     const RealisticChannelConfig& channelConfig) {

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        NS_LOG_WARN("Socket creation failed for AMC");
        return "";
    }

    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(9001);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        NS_LOG_WARN("AMC connection failed");
        close(sock);
        return "";
    }

    json request;
    request["type"] = "get_modulation";
    request["snr"] = std::clamp(snr, -5.0, 40.0);
    request["flow_id"] = flowId;
    request["timestamp"] = Simulator::Now().GetSeconds();

    request["channel_info"] = {
        {"type", channelConfig.channel_type},
        {"environment", "mixed"},
        {"mobility_speed", std::max(0.0, channelConfig.speed)},
        {"distance", std::max(1.0, channelConfig.distance)},
        {"carrier_frequency", channelConfig.carrier_frequency}
    };

    if (previousThroughput > 0 && !previousModulation.empty()) {
        request["feedback"] = {
            {"throughput", std::max(0.0, previousThroughput)},
            {"ber", std::clamp(previousBER, 0.0, 1.0)},
            {"bler", std::clamp(previousBLER, 0.0, 1.0)},
            {"modulation", previousModulation}
        };
    }

    std::string json_str = request.dump();

    ssize_t bytesSent = send(sock, json_str.c_str(), json_str.size(), 0);
    if (bytesSent <= 0) {
        NS_LOG_WARN("Failed to send to AMC");
        close(sock);
        return "";
    }

    char buffer[4096] = {0};
    ssize_t bytesRead = read(sock, buffer, 4096);
    close(sock);

    if (bytesRead <= 0) {
        NS_LOG_WARN("Empty AMC response");
        return "";
    }

    try {
        json response = json::parse(buffer);

        if (response.contains("modulation")) {
            std::string selectedModulation = response["modulation"].get<std::string>();

            if (selectedModulation == "qam4" || selectedModulation == "qam16" ||
                selectedModulation == "qam64" || selectedModulation == "qam256") {

                NS_LOG_INFO("AMC selected: " << selectedModulation
                           << " for SNR=" << snr << "dB, Flow=" << flowId);
                return selectedModulation;
            }
        }
    } catch (const std::exception& e) {
        NS_LOG_ERROR("AMC response parse error: " << e.what());
    }

    return "";
}

void SendFeedbackToAMC(uint32_t flowId, double throughput, double ber, double bler,
                       const std::string& modulation, const RealisticChannelConfig& channelConfig) {

    throughput = std::max(0.0, throughput);
    ber = std::clamp(ber, 0.0, 1.0);
    bler = std::clamp(bler, 0.0, 1.0);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return;

    struct timeval timeout;
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(9001);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sock);
        return;
    }

    json feedback_request = {
        {"type", "feedback_only"},
        {"flow_id", flowId},
        {"feedback", {
            {"throughput", throughput},
            {"ber", ber},
            {"bler", bler},
            {"modulation", modulation}
        }},
        {"channel_info", {
            {"type", channelConfig.channel_type},
            {"mobility_speed", channelConfig.speed},
            {"distance", channelConfig.distance},
            {"carrier_frequency", channelConfig.carrier_frequency}
        }}
    };

    std::string json_str = feedback_request.dump();
    send(sock, json_str.c_str(), json_str.size(), 0);
    close(sock);

    NS_LOG_INFO("Feedback AMC: Flow " << flowId
               << ", T=" << std::fixed << std::setprecision(1) << throughput << "Mbps"
               << ", BER=" << std::scientific << ber
               << ", BLER=" << std::fixed << std::setprecision(3) << bler
               << ", channel=" << channelConfig.channel_type);
}

void ProcessSionnaResponse(uint32_t flowId, double ber, double bler, uint32_t packetSizeBytes,
                          double delaySec, bool success, double throughputMbps) {
    auto& stats = realFlowStats[flowId];
    stats.txPackets++;
    stats.txBytes += packetSizeBytes;

    if (success && bler < 1.0) {
        stats.rxPackets++;

        double successRate = 1.0 - bler;
        uint64_t rxBytes = (uint64_t)(packetSizeBytes * successRate);
        stats.rxBytes += rxBytes;

        stats.totalDelay += delaySec;

        if (stats.packetsForDelay > 0) {
            stats.totalJitter += std::abs(delaySec - stats.lastPacketDelay);
        }
        stats.lastPacketDelay = delaySec;
        stats.packetsForDelay++;
    }

    stats.avgBER = (stats.avgBER * (stats.txPackets - 1) + ber) / stats.txPackets;
    stats.avgBLER = (stats.avgBLER * (stats.txPackets - 1) + bler) / stats.txPackets;
    stats.lastThroughput = throughputMbps;

    NS_LOG_INFO("Flow " << flowId << " - BER: " << std::scientific << ber
               << ", BLER: " << std::fixed << std::setprecision(3) << bler
               << ", Success: " << (success ? "yes" : "no")
               << ", Throughput: " << throughputMbps << " Mbps");
}

void SimulateTransmissionFallback(uint32_t flowId, int k, double snr_db,
                                 const std::string& modulation,
                                 const RealisticChannelConfig& channelConfig) {

    NS_LOG_INFO("Fallback simulation for Flow " << flowId);

    double snr_linear = std::pow(10.0, snr_db / 10.0);
    int modulationOrder = 4;

    if (modulation == "qam16") modulationOrder = 16;
    else if (modulation == "qam64") modulationOrder = 64;
    else if (modulation == "qam256") modulationOrder = 256;

    double ber;
    if (channelConfig.channel_type == "awgn") {
        ber = 0.5 * std::exp(-0.8 * snr_linear / modulationOrder);
    } else if (channelConfig.channel_type == "rayleigh") {
        ber = 0.5 / (1.0 + 0.8 * snr_linear / modulationOrder);
    } else if (channelConfig.channel_type == "urban") {
        ber = 0.7 / (1.0 + 0.6 * snr_linear / modulationOrder);
    } else {
        ber = 0.6 / (1.0 + 0.7 * snr_linear / modulationOrder);
    }

    double bler = std::min(ber * 30.0, 0.9);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.8, 1.2);
    ber *= dis(gen);
    bler *= dis(gen);

    ber = std::clamp(ber, 1e-8, 0.5);
    bler = std::clamp(bler, 0.0, 1.0);

    bool success = bler < 0.5;
    double throughput = success ? (k * 8.0 * (1.0 - bler) * 0.001) : 0.0;

    uint32_t packetSize = k / 8;
    double delay = Simulator::Now().GetSeconds();

    ProcessSionnaResponse(flowId, ber, bler, packetSize, delay, success, throughput);
    SendFeedbackToAMC(flowId, throughput, ber, bler, modulation, channelConfig);
}

void SendToSionnaWithDynamicChannel(uint32_t flowId, int k, double snr_db,
                                   const MobilityChannelData& mobility_data,
                                   double previousThroughput = 0.0,
                                   double previousBER = 0.0,
                                   double previousBLER = 0.0,
                                   std::string previousModulation = "") {

    if (k <= 0 || k > 10000) {
        NS_LOG_WARN("Invalid packet size: " << k << ", correcting to 64");
        k = 64;
    }

    if (snr_db < -10 || snr_db > 50) {
        NS_LOG_WARN("Invalid SNR: " << snr_db << "dB, correcting to 20dB");
        snr_db = 20.0;
    }

    RealisticChannelConfig channelConfig;
    ChannelEnvironment env = DetermineEnvironment(mobility_data.position,
                                                 global_mobility_config.area_width,
                                                 global_mobility_config.area_height);
    channelConfig = GetChannelConfigForEnvironment(env, mobility_data.distance_to_bs);
    channelConfig.speed = mobility_data.mobility_speed_kmh / 3.6;

    std::string selectedModulation = "";
    int amc_retries = 0;
    while (selectedModulation.empty() && amc_retries < 2) {
        selectedModulation = RequestModulationFromAMC(
            snr_db, flowId, previousThroughput, previousBER, previousBLER, previousModulation, channelConfig
        );
        if (selectedModulation.empty()) {
            amc_retries++;
            NS_LOG_WARN("AMC retry " << amc_retries << " for Flow " << flowId);
        }
    }

    if (selectedModulation.empty()) {
        if (snr_db < 10) selectedModulation = "qam4";
        else if (snr_db < 18) selectedModulation = "qam16";
        else if (snr_db < 25) selectedModulation = "qam64";
        else selectedModulation = "qam256";
        NS_LOG_WARN("AMC failed, using fallback: " << selectedModulation);
    }

    std::vector<int> bits;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < k; ++i) {
        bits.push_back(dis(gen));
    }

    json payload = {
        {"id", static_cast<int>(flowId)},
        {"k", k},
        {"modulation", selectedModulation},
        {"snr_db", snr_db},
        {"payload", json::array()},
        {"channel_type", channelConfig.channel_type},
        {"channel_params", {
            {"carrier_frequency", channelConfig.carrier_frequency},
            {"speed", channelConfig.speed},
            {"distance", channelConfig.distance},
            {"num_paths", channelConfig.num_paths},
            {"delay_spread", channelConfig.delay_spread},
            {"k_factor", channelConfig.k_factor},
            {"phase_noise_std", channelConfig.phase_noise_std}
        }},
        {"mobility_context", {
            {"position", {mobility_data.position.x, mobility_data.position.y, mobility_data.position.z}},
            {"velocity", {mobility_data.velocity.x, mobility_data.velocity.y, mobility_data.velocity.z}},
            {"environment_type", mobility_data.environment_type},
            {"mobility_speed_kmh", mobility_data.mobility_speed_kmh},
            {"channel_stability", mobility_data.channel_stability},
            {"channel_condition", mobility_data.channel_condition},
            {"timestamp", mobility_data.timestamp}
        }}
    };

    for (int bit : bits) {
        payload["payload"].push_back(static_cast<float>(bit));
    }

    std::string json_str = payload.dump();

    bool sionna_success = false;
    double ber = 0.0, bler = 0.0, throughput = 0.0;
    bool transmission_success = false;
    bool use_sionna_throughput = false;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock >= 0) {
        struct timeval timeout;
        timeout.tv_sec = 8;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(9000);
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >= 0) {
            ssize_t bytesSent = send(sock, json_str.c_str(), json_str.size(), 0);

            if (bytesSent > 0) {
                char buffer[8192] = {0};
                ssize_t bytesRead = read(sock, buffer, 8192);

                if (bytesRead > 0) {
                    try {
                        json response = json::parse(buffer);

                        if (response.contains("ber") && response.contains("bler")) {
                            ber = response["ber"].get<double>();
                            bler = response["bler"].get<double>();
                            transmission_success = response.value("success", true);

                            if (response.contains("effective_throughput")) {
                                throughput = response["effective_throughput"].get<double>();
                                use_sionna_throughput = true;
                                NS_LOG_INFO("Using Sionna effective_throughput: " << throughput);
                            } else if (response.contains("throughput_mbps")) {
                                throughput = response["throughput_mbps"].get<double>();
                                use_sionna_throughput = true;
                                NS_LOG_INFO("Using Sionna throughput_mbps: " << throughput);
                            } else if (response.contains("throughput")) {
                                throughput = response["throughput"].get<double>();
                                use_sionna_throughput = true;
                                NS_LOG_INFO("Using Sionna throughput: " << throughput);
                            }

                            if (!use_sionna_throughput || throughput <= 0) {
                                if (transmission_success) {
                                    throughput = (k * 8.0 * (1.0 - bler)) / 1000.0;
                                    NS_LOG_WARN("Sionna throughput not available, using fallback: " << throughput);
                                } else {
                                    throughput = 0.0;
                                }
                            }

                            sionna_success = true;
                            NS_LOG_INFO("Sionna success: Flow " << flowId
                                       << " BER=" << std::scientific << ber
                                       << " BLER=" << std::fixed << std::setprecision(3) << bler
                                       << " T=" << throughput << "Mbps"
                                       << " (Sionna=" << (use_sionna_throughput ? "YES" : "NO") << ")");
                        } else {
                            NS_LOG_WARN("Sionna response missing BER/BLER fields");
                        }
                    } catch (const std::exception& e) {
                        NS_LOG_ERROR("JSON parse error: " << e.what());
                        sionna_success = false;
                    }
                } else {
                    NS_LOG_WARN("No response from Sionna");
                }
            } else {
                NS_LOG_WARN("Failed to send to Sionna");
            }
        } else {
            NS_LOG_WARN("Cannot connect to Sionna");
        }
        close(sock);
    } else {
        NS_LOG_WARN("Cannot create socket for Sionna");
    }

    if (!sionna_success) {
        NS_LOG_WARN("Sionna failed, using fallback simulation for Flow " << flowId);

        double snr_linear = std::pow(10.0, snr_db / 10.0);
        int modulationOrder = (selectedModulation == "qam4") ? 4 :
                             (selectedModulation == "qam16") ? 16 :
                             (selectedModulation == "qam64") ? 64 : 256;

        if (channelConfig.channel_type == "awgn") {
            ber = 0.5 * std::exp(-0.8 * snr_linear / modulationOrder);
        } else if (channelConfig.channel_type == "rayleigh") {
            ber = 0.5 / (1.0 + 0.8 * snr_linear / modulationOrder);
        } else if (channelConfig.channel_type == "urban") {
            ber = 0.7 / (1.0 + 0.6 * snr_linear / modulationOrder);
        } else {
            ber = 0.6 / (1.0 + 0.7 * snr_linear / modulationOrder);
        }

        bler = std::min(ber * 30.0, 0.9);

        std::uniform_real_distribution<> dis(0.8, 1.2);
        ber *= dis(gen);
        bler *= dis(gen);

        transmission_success = bler < 0.5;
        throughput = transmission_success ? (k * 8.0 * (1.0 - bler) * 0.001) : 0.0;
        use_sionna_throughput = false;
    }

    ber = std::clamp(ber, 1e-8, 0.5);
    bler = std::clamp(bler, 0.0, 1.0);
    throughput = std::max(0.0, throughput);

    bool data_updated = false;
    for (auto it = mobility_data_log.rbegin(); it != mobility_data_log.rend(); ++it) {
        if (it->flowId == flowId &&
            std::abs(it->timestamp - Simulator::Now().GetSeconds()) < 0.2) {
            it->ber = ber;
            it->bler = bler;
            it->modulation = selectedModulation;
            it->throughput_mbps = throughput;
            it->transmission_success = transmission_success;
            data_updated = true;
            break;
        }
    }

    if (!data_updated) {
        MobilityChannelData new_entry = mobility_data;
        new_entry.ber = ber;
        new_entry.bler = bler;
        new_entry.modulation = selectedModulation;
        new_entry.throughput_mbps = throughput;
        new_entry.transmission_success = transmission_success;
        new_entry.timestamp = Simulator::Now().GetSeconds();
        mobility_data_log.push_back(new_entry);

        NS_LOG_INFO("Created new mobility entry for Flow " << flowId
                   << " with throughput=" << throughput
                   << " (Sionna=" << (use_sionna_throughput ? "YES" : "NO") << ")");
    }

    SendFeedbackToAMC(flowId, throughput, ber, bler, selectedModulation, channelConfig);

    uint32_t packetSize = k / 8;
    double delay = Simulator::Now().GetSeconds();
    ProcessSionnaResponse(flowId, ber, bler, packetSize, delay, transmission_success, throughput);

    if (flowHistories.find(flowId) != flowHistories.end()) {
        flowHistories[flowId].lastThroughput = throughput;
        flowHistories[flowId].lastBER = ber;
        flowHistories[flowId].lastBLER = bler;
        flowHistories[flowId].lastModulation = selectedModulation;
        flowHistories[flowId].channelConfig = channelConfig;
    }
}

void CollectMobilityChannelData() {
    Time current_time = Simulator::Now();
    double timestamp = current_time.GetSeconds();

    for (uint32_t flow_id = 0; flow_id < 10; ++flow_id) {

        Vector bs_position(500.0, 500.0, 10.0);

        Vector ue_position;
        ue_position.x = 200 + 300 * sin(timestamp * 0.1 + flow_id);
        ue_position.y = 200 + 200 * cos(timestamp * 0.15 + flow_id * 0.5);
        ue_position.z = 1.5;

        Vector velocity(0, 0, 0);
        if (previous_positions.find(flow_id) != previous_positions.end() &&
            last_update_time.find(flow_id) != last_update_time.end()) {

            Time dt = current_time - last_update_time[flow_id];
            if (dt.GetSeconds() > 0) {
                velocity.x = (ue_position.x - previous_positions[flow_id].x) / dt.GetSeconds();
                velocity.y = (ue_position.y - previous_positions[flow_id].y) / dt.GetSeconds();
                velocity.z = (ue_position.z - previous_positions[flow_id].z) / dt.GetSeconds();
            }
        }

        double snr_db = CalculateRealisticSNR(ue_position, bs_position, timestamp, flow_id, global_channel_config);

        if (snr_history[flow_id].size() >= 20) {
            snr_history[flow_id].pop_front();
        }
        snr_history[flow_id].push_back(snr_db);

        double distance_to_bs = CalculateDistance(ue_position, bs_position);
        double mobility_speed_kmh = sqrt(velocity.x*velocity.x + velocity.y*velocity.y) * 3.6;
        double channel_stability = CalculateChannelStability(flow_id);

        ChannelEnvironment env = DetermineEnvironment(ue_position,
                                                     global_mobility_config.area_width,
                                                     global_mobility_config.area_height);
        std::string environment_type;
        switch(env) {
            case INDOOR: environment_type = "indoor"; break;
            case URBAN: environment_type = "urban"; break;
            case SUBURBAN: environment_type = "suburban"; break;
            case RURAL: environment_type = "rural"; break;
        }

        std::string channel_condition = (distance_to_bs < 200.0 && env != INDOOR) ? "LOS" : "NLOS";

        MobilityChannelData data;
        data.flowId = flow_id;
        data.timestamp = timestamp;
        data.position = ue_position;
        data.velocity = velocity;
        data.distance_to_bs = distance_to_bs;
        data.snr_db = snr_db;
        data.pathloss_db = 20 * log10(distance_to_bs) + 20 * log10(global_channel_config.carrier_frequency / 1e9);
        data.channel_condition = channel_condition;
        data.environment_type = environment_type;
        data.mobility_speed_kmh = mobility_speed_kmh;
        data.channel_stability = channel_stability;
        data.ber = 0.0;
        data.bler = 0.0;
        data.modulation = "";
        data.throughput_mbps = 0.0;
        data.transmission_success = false;

        mobility_data_log.push_back(data);

        previous_positions[flow_id] = ue_position;
        last_update_time[flow_id] = current_time;

        if (flow_id == 0 && (int(timestamp * 10) % 50) == 0) {
            NS_LOG_INFO("Flow " << flow_id << " @ t=" << timestamp << "s: "
                       << "pos=(" << ue_position.x << "," << ue_position.y << "), "
                       << "SNR=" << snr_db << "dB, "
                       << "speed=" << mobility_speed_kmh << "km/h, "
                       << "env=" << environment_type);
        }
    }
}

void ExportDataForGAN(const std::string& filename = "mobility_channel_data_enhanced.json") {

    NS_LOG_INFO("Exporting GAN data: " << mobility_data_log.size() << " samples");

    json output;
    output["metadata"] = {
        {"total_samples", mobility_data_log.size()},
        {"simulation_duration", Simulator::Now().GetSeconds()},
        {"channel_config", {
            {"type", global_channel_config.channel_type},
            {"frequency_ghz", global_channel_config.carrier_frequency / 1e9},
            {"shadowing_std", global_channel_config.shadowing_std},
            {"coherence_time", global_channel_config.coherence_time}
        }},
        {"mobility_config", {
            {"type", global_mobility_config.type},
            {"max_speed_kmh", global_mobility_config.max_speed_kmh},
            {"area_size", {global_mobility_config.area_width, global_mobility_config.area_height}}
        }},
        {"amc_integration", true},
        {"sionna_integration", true},
        {"export_timestamp", time(nullptr)}
    };

    std::map<uint32_t, json> flows_data;

    for (const auto& data : mobility_data_log) {
        json sample = {
            {"timestamp", data.timestamp},
            {"position", {data.position.x, data.position.y, data.position.z}},
            {"velocity", {data.velocity.x, data.velocity.y, data.velocity.z}},
            {"distance_to_bs", data.distance_to_bs},
            {"snr_db", data.snr_db},
            {"pathloss_db", data.pathloss_db},
            {"channel_condition", data.channel_condition},
            {"environment_type", data.environment_type},
            {"mobility_speed_kmh", data.mobility_speed_kmh},
            {"channel_stability", data.channel_stability},
            {"ber", data.ber},
            {"bler", data.bler},
            {"modulation", data.modulation},
            {"throughput_mbps", data.throughput_mbps},
            {"transmission_success", data.transmission_success}
        };

        flows_data[data.flowId]["samples"].push_back(sample);
    }

    output["flows"] = flows_data;

    if (!mobility_data_log.empty()) {
        std::vector<double> all_snr, all_speeds, all_distances, all_ber, all_bler, all_throughput;

        for (const auto& data : mobility_data_log) {
            all_snr.push_back(data.snr_db);
            all_speeds.push_back(data.mobility_speed_kmh);
            all_distances.push_back(data.distance_to_bs);
            if (data.ber > 0) all_ber.push_back(data.ber);
            if (data.bler > 0) all_bler.push_back(data.bler);
            if (data.throughput_mbps > 0) all_throughput.push_back(data.throughput_mbps);
        }

        auto calc_stats = [](const std::vector<double>& values) {
            if (values.empty()) return json{{"mean", 0}, {"std", 0}, {"min", 0}, {"max", 0}};

            double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
            double std = sqrt(sq_sum / values.size() - mean * mean);
            double min_val = *std::min_element(values.begin(), values.end());
            double max_val = *std::max_element(values.begin(), values.end());

            return json{
                {"mean", mean},
                {"std", std},
                {"min", min_val},
                {"max", max_val}
            };
        };

        output["statistics"] = {
            {"snr_db", calc_stats(all_snr)},
            {"mobility_speed_kmh", calc_stats(all_speeds)},
            {"distance_to_bs_m", calc_stats(all_distances)},
            {"ber", calc_stats(all_ber)},
            {"bler", calc_stats(all_bler)},
            {"throughput_mbps", calc_stats(all_throughput)}
        };
    }

    std::ofstream file(filename);
    if (file.is_open()) {
        file << output.dump(2);
        file.close();
        NS_LOG_INFO("Data exported to: " << filename);
    } else {
        NS_LOG_ERROR("Cannot open file: " << filename);
    }
}

void SetupDynamicMobility(const NodeContainer& ueNodes, const DynamicMobilityConfig& config) {

    NS_LOG_INFO("Setting up dynamic mobility: type=" << config.type);

    MobilityHelper mobility;

    switch (config.type) {
        case URBAN_GRID: {
            mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                    "Bounds", RectangleValue(Rectangle(0, config.area_width, 0, config.area_height)),
                                    "Speed", StringValue("ns3::UniformRandomVariable[Min=" +
                                                       std::to_string(config.min_speed_kmh/3.6) +
                                                       "|Max=" + std::to_string(config.max_speed_kmh/3.6) + "]"),
                                    "Direction", StringValue("ns3::UniformRandomVariable[Min=0|Max=6.28318]"));
            break;
        }

        case HIGHWAY: {
            mobility.SetMobilityModel("ns3::SteadyStateRandomWaypointMobilityModel",
                                    "MinSpeed", DoubleValue(config.min_speed_kmh/3.6),
                                    "MaxSpeed", DoubleValue(config.max_speed_kmh/3.6));
            break;
        }

        case CIRCULAR: {
            mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                    "Mode", StringValue("Time"),
                                    "Time", StringValue("5s"),
                                    "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" +
                                                       std::to_string((config.min_speed_kmh + config.max_speed_kmh)/2/3.6) + "]"));
            break;
        }

        default:
            mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel");
    }

    mobility.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
                                 "X", DoubleValue(config.area_width/2),
                                 "Y", DoubleValue(config.area_height/2),
                                 "rho", DoubleValue(std::min(config.area_width, config.area_height)/4));

    mobility.Install(ueNodes);

    NS_LOG_INFO("Mobility installed on " << ueNodes.GetN() << " UEs");
}

void ScheduleDynamicTransmissions() {
    CollectMobilityChannelData();

    double current_time = Simulator::Now().GetSeconds();

    for (uint32_t flowId = 0; flowId < 10; ++flowId) {
        MobilityChannelData* current_data = nullptr;
        for (auto it = mobility_data_log.rbegin(); it != mobility_data_log.rend(); ++it) {
            if (it->flowId == flowId && (current_time - it->timestamp) < 0.15) {
                current_data = &(*it);
                break;
            }
        }

        if (current_data != nullptr) {
            double prevThroughput = 0.0;
            double prevBER = 0.0;
            double prevBLER = 0.0;
            std::string prevModulation = "";

            if (flowHistories.find(flowId) != flowHistories.end()) {
                const auto& history = flowHistories[flowId];
                prevThroughput = history.lastThroughput;
                prevBER = history.lastBER;
                prevBLER = history.lastBLER;
                prevModulation = history.lastModulation;
            }

            Simulator::Schedule(MilliSeconds(10), &SendToSionnaWithDynamicChannel,
                              flowId, 64, current_data->snr_db, *current_data,
                              prevThroughput, prevBER, prevBLER, prevModulation);
        } else {
            NS_LOG_WARN("No recent mobility data for Flow " << flowId);
        }
    }
}

void PrintRealisticFlowStats() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== NS-3 + SIONNA + AMC FLOW METRICS ===\n";
    double totalThroughput = 0.0;
    double totalDelayAvg = 0.0;
    int validFlows = 0;

    for (const auto& [flowId, stats] : realFlowStats) {
        if (stats.txPackets == 0) continue;

        double offeredMbps = (stats.txBytes * 8.0) / simTime / 1e6;
        double throughputMbps = (stats.rxBytes * 8.0) / simTime / 1e6;
        double meanDelay = stats.packetsForDelay > 0 ? stats.totalDelay / stats.packetsForDelay : 0;
        double meanJitter = stats.packetsForDelay > 0 ? stats.totalJitter / stats.packetsForDelay : 0;
        double packetLossRate = 1.0 - (double(stats.rxPackets) / double(stats.txPackets));

        totalThroughput += throughputMbps;
        totalDelayAvg += meanDelay;
        validFlows++;

        std::cout << "Flow " << flowId << ":\n";
        std::cout << "  Tx Packets: " << stats.txPackets << "\n";
        std::cout << "  Rx Packets: " << stats.rxPackets << " (" << (100.0 * stats.rxPackets / stats.txPackets) << "%)\n";
        std::cout << "  Packet Loss: " << std::setprecision(1) << (packetLossRate * 100) << "%\n";
        std::cout << "  TxOffered:  " << std::setprecision(2) << offeredMbps << " Mbps\n";
        std::cout << "  Throughput: " << std::setprecision(2) << throughputMbps << " Mbps\n";
        std::cout << "  Mean delay: " << std::setprecision(2) << meanDelay * 1000 << " ms\n";
        std::cout << "  Avg BER:    " << std::scientific << stats.avgBER << "\n";
        std::cout << "  Avg BLER:   " << std::fixed << std::setprecision(3) << stats.avgBLER << "\n";
        std::cout << "  Last Mod:   " << stats.lastModulation << "\n\n";
    }

    if (validFlows > 0) {
        std::cout << "=== SUMMARY ===\n";
        std::cout << "  Active flows: " << validFlows << "\n";
        std::cout << "  Avg throughput: " << std::setprecision(2) << totalThroughput / validFlows << " Mbps\n";
        std::cout << "  Avg delay: " << std::setprecision(2) << totalDelayAvg / validFlows * 1000 << " ms\n";
        std::cout << "  Simulation time: " << simTime << " s\n";
        std::cout << "  Collected samples: " << mobility_data_log.size() << "\n";
    }

    std::cout << "========================================\n";
}

int main(int argc, char* argv[])
{
    uint16_t gNbNum = 1;
    uint16_t ueNumPergNb = 10;
    bool logging = false;
    bool doubleOperationalBand = false;

    uint32_t udpPacketSizeULL = 100;
    uint32_t udpPacketSizeBe = 1252;
    uint32_t lambdaULL = 10000;
    uint32_t lambdaBe = 10000;

    Time simTimeNs3 = Seconds(60);
    Time udpAppStartTime = Seconds(5);

    uint16_t numerologyBwp1 = 4;
    double centralFrequencyBand1 = 28e9;
    double bandwidthBand1 = 50e6;
    double totalTxPower = 35;

    global_mobility_config.type = URBAN_GRID;
    global_mobility_config.max_speed_kmh = 80.0;
    global_mobility_config.min_speed_kmh = 5.0;
    global_mobility_config.area_width = 2000.0;
    global_mobility_config.area_height = 2000.0;

    uint32_t mobilityType = 5;

    global_channel_config.channel_type = "urban";
    global_channel_config.carrier_frequency = centralFrequencyBand1;
    global_channel_config.enable_temporal_correlation = true;

    std::string simTag = "dynamic-enhanced";
    std::string outputDir = "./";
    std::string ganDataFile = "gan_data_.json";

    CommandLine cmd(__FILE__);
    cmd.AddValue("gNbNum", "Number of gNBs", gNbNum);
    cmd.AddValue("ueNumPergNb", "Number of UE per gNB", ueNumPergNb);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.AddValue("simTime", "Simulation time", simTimeNs3);
    cmd.AddValue("maxSpeed", "Maximum UE speed (km/h)", global_mobility_config.max_speed_kmh);
    cmd.AddValue("minSpeed", "Minimum UE speed (km/h)", global_mobility_config.min_speed_kmh);
    cmd.AddValue("areaWidth", "Simulation area width (m)", global_mobility_config.area_width);
    cmd.AddValue("areaHeight", "Simulation area height (m)", global_mobility_config.area_height);
    cmd.AddValue("channelType", "Channel type (urban, suburban, rural)", global_channel_config.channel_type);
    cmd.AddValue("ganDataFile", "Output file for GAN training data", ganDataFile);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.AddValue("mobilityType", "Mobility type (0=STATIC, 1=LINEAR, 2=CIRCULAR, 3=RANDOM_WALK, 4=HIGHWAY, 5=URBAN_GRID)", mobilityType);

    cmd.Parse(argc, argv);
    global_mobility_config.type = static_cast<MobilityType>(mobilityType);

    simTime = simTimeNs3.GetSeconds();

    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 && centralFrequencyBand1 > 100e9);

    if (logging) {
        LogComponentEnable("CttcNrDemoDynamicEnhanced", LOG_LEVEL_INFO);
        LogComponentEnable("UdpClient", LOG_LEVEL_INFO);
        LogComponentEnable("UdpServer", LOG_LEVEL_INFO);
    }

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    int64_t randomStream = 1;
    GridScenarioHelper gridScenario;
    gridScenario.SetRows(1);
    gridScenario.SetColumns(gNbNum);
    gridScenario.SetHorizontalBsDistance(global_mobility_config.area_width / std::max(1, static_cast<int>(gNbNum)));
    gridScenario.SetVerticalBsDistance(global_mobility_config.area_height / 2);
    gridScenario.SetBsHeight(25);
    gridScenario.SetUtHeight(1.5);
    gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber(gNbNum);
    gridScenario.SetUtNumber(ueNumPergNb * gNbNum);
    gridScenario.SetScenarioHeight(global_mobility_config.area_height / 100);
    gridScenario.SetScenarioLength(global_mobility_config.area_width / 100);
    randomStream += gridScenario.AssignStreams(randomStream);
    gridScenario.CreateScenario();

    NS_LOG_INFO("Created " << gridScenario.GetUserTerminals().GetN() << " UEs and "
                << gridScenario.GetBaseStations().GetN() << " gNBs");

    SetupDynamicMobility(gridScenario.GetUserTerminals(), global_mobility_config);

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                    bandwidthBand1,
                                                    numCcPerBand//,
                                                    // BandwidthPartInfo::UMi_StreetCanyon
                                                );
            

    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);

    // Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(100)));
    // nrHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(100)));
    // nrHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    // nrHelper->InitializeOperationBand(&band1);

    // Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>(); 
    //     channelHelper->ConfigureFactories(
    //     scenario,
    //     "Default",
    //     "ThreeGpp"); // Configure the spectrum channel with the scenario
    // channelHelper->AssignChannelsToBands({band1});

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    /**
     * Use channelHelper API to define the attributes for the channel model (condition, pathloss and
     * spectrum)
     */
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(100)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    channelHelper->AssignChannelsToBands({band1});
    allBwps = CcBwpCreator::GetAllBwps({band1});
    
    double x = pow(10, totalTxPower / 10);





    Packet::EnableChecking();
    Packet::EnablePrinting();

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gridScenario.GetBaseStations(), allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(gridScenario.GetUserTerminals(), allBwps);

    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueNetDev, randomStream);

    for (uint32_t i = 0; i < gnbNetDev.GetN(); ++i) {
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        nrHelper->GetGnbPhy(gnbNetDev.Get(i), 0)->SetAttribute("TxPower", DoubleValue(10 * log10(x)));
    }

    nrHelper->UpdateDeviceConfigs(gnbNetDev);
    nrHelper->UpdateDeviceConfigs(ueNetDev);

    Ptr<Node> pgw = nrEpcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.000)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    internet.Install(gridScenario.GetUserTerminals());

    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));

    for (uint32_t j = 0; j < gridScenario.GetUserTerminals().GetN(); ++j) {
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(
            gridScenario.GetUserTerminals().Get(j)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(nrEpcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);

    uint16_t dlPort = 1234;
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    UdpServerHelper dlPacketSink(dlPort);
    serverApps.Add(dlPacketSink.Install(gridScenario.GetUserTerminals()));

    UdpClientHelper dlClient;
    dlClient.SetAttribute("RemotePort", UintegerValue(dlPort));
    dlClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClient.SetAttribute("PacketSize", UintegerValue(udpPacketSizeULL));
    dlClient.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaULL)));

    NrEpsBearer bearer(NrEpsBearer::NGBR_LOW_LAT_EMBB);
    Ptr<NrEpcTft> tft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter dlpf;
    dlpf.localPortStart = dlPort;
    dlpf.localPortEnd = dlPort;
    tft->Add(dlpf);

    for (uint32_t i = 0; i < gridScenario.GetUserTerminals().GetN(); ++i) {
        Ptr<NetDevice> ueDevice = ueNetDev.Get(i);
        Address ueAddress = ueIpIface.GetAddress(i);

        dlClient.SetAttribute("RemoteAddress", AddressValue(ueAddress));
        clientApps.Add(dlClient.Install(remoteHost));

        nrHelper->ActivateDedicatedEpsBearer(ueDevice, bearer, tft);
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTimeNs3);
    clientApps.Stop(simTimeNs3);

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(gridScenario.GetUserTerminals());

    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    for (uint32_t i = 0; i < gridScenario.GetUserTerminals().GetN(); ++i) {
        flowHistories[i] = FlowHistory();

        Ptr<Node> ue = gridScenario.GetUserTerminals().Get(i);
        Ptr<MobilityModel> mobility = ue->GetObject<MobilityModel>();
        Vector initialPosition = mobility->GetPosition();

        flowHistories[i].environment = DetermineEnvironment(initialPosition,
                                                           global_mobility_config.area_width,
                                                           global_mobility_config.area_height);

        flowHistories[i].channelConfig = GetChannelConfigForEnvironment(
            flowHistories[i].environment,
            CalculateDistance(initialPosition, Vector(500.0, 500.0, 10.0))
        );
        flowHistories[i].channelConfig.carrier_frequency = centralFrequencyBand1;

        NS_LOG_INFO("Flow " << i << " initialized: env="
                   << (flowHistories[i].environment == INDOOR ? "indoor" :
                       flowHistories[i].environment == URBAN ? "urban" :
                       flowHistories[i].environment == SUBURBAN ? "suburban" : "rural")
                   << ", channel=" << flowHistories[i].channelConfig.channel_type);
    }

    double data_collection_interval = 0.1;
    for (double t = 1.0; t < simTimeNs3.GetSeconds(); t += data_collection_interval) {
        Simulator::Schedule(Seconds(t), &ScheduleDynamicTransmissions);
    }

    NS_LOG_INFO("Starting simulation - Duration: " << simTimeNs3.GetSeconds()
               << "s, UEs: " << gridScenario.GetUserTerminals().GetN()
               << ", Area: " << global_mobility_config.area_width << "x" << global_mobility_config.area_height << "m");

    Simulator::Stop(simTimeNs3);
    Simulator::Run();

    NS_LOG_INFO("Simulation complete. Exporting GAN data...");

    std::string full_gan_data_path = outputDir + "/" + ganDataFile;
    ExportDataForGAN(full_gan_data_path);

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    std::string statsFile = outputDir + "/" + simTag + "_flow_stats.txt";
    std::ofstream outFile;
    outFile.open(statsFile.c_str(), std::ofstream::out | std::ofstream::trunc);

    if (outFile.is_open()) {
        outFile.setf(std::ios_base::fixed);

        double flowDuration = (simTimeNs3 - udpAppStartTime).GetSeconds();
        double averageFlowThroughput = 0.0;
        double averageFlowDelay = 0.0;
        int validFlows = 0;

        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin();
             i != stats.end(); ++i) {

            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);

            outFile << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort
                    << " -> " << t.destinationAddress << ":" << t.destinationPort << ")\n";
            outFile << "  Tx Packets: " << i->second.txPackets << "\n";
            outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
            outFile << "  TxOffered:  " << i->second.txBytes * 8.0 / flowDuration / 1000.0 / 1000.0 << " Mbps\n";
            outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";

            if (i->second.rxPackets > 0) {
                double throughput = i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
                double meanDelay = 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets;

                averageFlowThroughput += throughput;
                averageFlowDelay += meanDelay;
                validFlows++;

                outFile << "  Throughput: " << throughput << " Mbps\n";
                outFile << "  Mean delay: " << meanDelay << " ms\n";
                outFile << "  Mean jitter: " << 1000 * i->second.jitterSum.GetSeconds() / i->second.rxPackets << " ms\n";
            } else {
                outFile << "  Throughput: 0 Mbps\n";
                outFile << "  Mean delay: 0 ms\n";
                outFile << "  Mean jitter: 0 ms\n";
            }
            outFile << "  Rx Packets: " << i->second.rxPackets << "\n\n";
        }

        if (validFlows > 0) {
            double meanFlowThroughput = averageFlowThroughput / validFlows;
            double meanFlowDelay = averageFlowDelay / validFlows;

            outFile << "\nSUMMARY:\n";
            outFile << "  Valid flows: " << validFlows << "\n";
            outFile << "  Mean flow throughput: " << meanFlowThroughput << " Mbps\n";
            outFile << "  Mean flow delay: " << meanFlowDelay << " ms\n";
            outFile << "  Total mobility samples: " << mobility_data_log.size() << "\n";
            outFile << "  Data collection rate: " << mobility_data_log.size() / simTime << " samples/s\n";
            outFile << "  AMC integration: ENABLED\n";
            outFile << "  Sionna integration: ENABLED\n";
            outFile << "  Realistic channels: ENABLED\n";
        }

        outFile.close();

        std::ifstream f(statsFile.c_str());
        if (f.is_open()) {
            std::cout << f.rdbuf();
        }
    }

    NS_LOG_INFO("Simulation finished");
    NS_LOG_INFO("Mobility samples collected: " << mobility_data_log.size());
    NS_LOG_INFO("GAN data file: " << full_gan_data_path);
    NS_LOG_INFO("Flow stats: " << statsFile);

    if (mobility_data_log.size() > 0) {
        std::vector<double> all_snr, all_speeds, all_distances, all_ber, all_bler, all_throughput;
        std::map<std::string, int> env_count, mod_count;
        int successful_transmissions = 0;

        for (const auto& data : mobility_data_log) {
            all_snr.push_back(data.snr_db);
            all_speeds.push_back(data.mobility_speed_kmh);
            all_distances.push_back(data.distance_to_bs);
            env_count[data.environment_type]++;

            if (data.ber > 0) all_ber.push_back(data.ber);
            if (data.bler > 0) all_bler.push_back(data.bler);
            if (data.throughput_mbps > 0) all_throughput.push_back(data.throughput_mbps);
            if (!data.modulation.empty()) mod_count[data.modulation]++;
            if (data.transmission_success) successful_transmissions++;
        }

        auto minmax_snr = std::minmax_element(all_snr.begin(), all_snr.end());
        auto minmax_speed = std::minmax_element(all_speeds.begin(), all_speeds.end());

        NS_LOG_INFO("SNR range: [" << *minmax_snr.first << ", " << *minmax_snr.second << "] dB");
        NS_LOG_INFO("Speed range: [" << *minmax_speed.first << ", " << *minmax_speed.second << "] km/h");

        if (!all_ber.empty()) {
            double avg_ber = std::accumulate(all_ber.begin(), all_ber.end(), 0.0) / all_ber.size();
            NS_LOG_INFO("Avg BER: " << std::scientific << avg_ber);
        }

        if (!all_bler.empty()) {
            double avg_bler = std::accumulate(all_bler.begin(), all_bler.end(), 0.0) / all_bler.size();
            NS_LOG_INFO("Avg BLER: " << std::fixed << avg_bler);
        }

        if (!all_throughput.empty()) {
            double avg_throughput = std::accumulate(all_throughput.begin(), all_throughput.end(), 0.0) / all_throughput.size();
            NS_LOG_INFO("Avg throughput: " << avg_throughput << " Mbps");
        }

        NS_LOG_INFO("Successful transmissions: " << successful_transmissions << "/" << mobility_data_log.size()
                   << " (" << (100.0 * successful_transmissions / mobility_data_log.size()) << "%)");

        NS_LOG_INFO("Environment distribution:");
        for (const auto& [env, count] : env_count) {
            NS_LOG_INFO("  " << env << ": " << count << " samples");
        }

        NS_LOG_INFO("Modulation distribution:");
        for (const auto& [mod, count] : mod_count) {
            NS_LOG_INFO("  " << mod << ": " << count << " uses");
        }

        NS_LOG_INFO("Simulation duration: " << simTime << " seconds");
    }

    PrintRealisticFlowStats();

    Simulator::Destroy();
    return 0;
}
