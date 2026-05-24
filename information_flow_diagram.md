# Information Flow Diagram — Metrics Server, AMC, Sionna & NS-3

## High-Level Architecture

```mermaid
flowchart TB
    subgraph External["External Traffic Source"]
        NS3["NS-3 Simulation<br/><i>or Benchmark Client</i><br/>(send_modulation_requests.py)"]
    end

    subgraph AMC_System["AMC Decision Engine — :9001"]
        AMC["IntegratedAMCServer<br/>(amc.py)"]
        GAN["GAN SNR Predictor<br/>(gan_snr_predictor.py)"]
        RL["RL Agent / Table Lookup<br/>(amc_server.py)"]
        PA["Performance Analyzer<br/>(performance_analyzer.py)"]
    end

    subgraph PHY["Physical Layer Simulator — :9000"]
        SIONNA["Sionna Server<br/>(sionna_server.py)<br/>TensorFlow + Sionna PHY"]
    end

    subgraph Observability["Observability Pipeline — :5001 / :5000"]
        EP["Event Publisher<br/>(event_publisher.py)"]
        MS["Metrics Server<br/>(metrics_server.py)<br/>Flask REST API"]
        DASH["Dashboard<br/>(dashboard.py)<br/>Flask + Chart.js"]
        BENCH["Benchmark Client<br/>(run_benchmark_client.py)"]
        PLOT["Plot Figures<br/>(plot_figures.py)"]
    end

    NS3 -->|"① TCP :9001<br/>get_modulation request<br/>{snr, flow_id, channel_info}"| AMC
    AMC -->|"② TCP :9000<br/>PHY simulation request<br/>{modulation, snr_db, payload,<br/>channel_type, channel_params}"| SIONNA
    SIONNA -->|"③ TCP response<br/>{ber, bler, effective_throughput,<br/>success, processing_time_ms}"| AMC
    AMC -->|"④ TCP response<br/>{modulation, method,<br/>transmission_result,<br/>decision_info, performance}"| NS3

    AMC --- GAN
    AMC --- RL
    AMC --- PA

    AMC -->|"⑤ Async queue"| EP
    EP -->|"⑥ HTTP POST :5001<br/>/ingest_event<br/>{snr, ber, bler, throughput,<br/>decision_method, latencies}"| MS

    MS -->|"⑦ HTTP GET<br/>/api/events"| DASH
    MS -->|"⑧ HTTP GET<br/>/api/events"| BENCH
    MS -->|"⑨ HTTP GET<br/>/api/events"| PLOT

    style NS3 fill:#2d3748,stroke:#4299e1,color:#fff
    style AMC fill:#2b6cb0,stroke:#63b3ed,color:#fff
    style GAN fill:#805ad5,stroke:#b794f4,color:#fff
    style RL fill:#38a169,stroke:#68d391,color:#fff
    style PA fill:#d69e2e,stroke:#ecc94b,color:#000
    style SIONNA fill:#c53030,stroke:#fc8181,color:#fff
    style EP fill:#718096,stroke:#a0aec0,color:#fff
    style MS fill:#dd6b20,stroke:#f6ad55,color:#fff
    style DASH fill:#d53f8c,stroke:#f687b3,color:#fff
    style BENCH fill:#319795,stroke:#4fd1c5,color:#fff
    style PLOT fill:#319795,stroke:#4fd1c5,color:#fff
```

---

## Detailed Request-Response Flow

### ① NS-3 / Client → AMC Server (TCP :9001)

The ns-3 simulation (or test tools like `send_modulation_requests.py`) opens a raw TCP connection to the AMC server and sends a JSON request:

```json
{
  "type": "get_modulation",
  "flow_id": 1,
  "snr": 18.5,
  "channel_info": {
    "type": "rayleigh",
    "mobility_speed": 3.0,
    "distance": 100.0,
    "carrier_frequency": 2.4e9
  }
}
```

Other supported request types: `feedback` / `feedback_only`, `get_stats`.

---

### ② ③ AMC Server ↔ Sionna Server (TCP :9000)

For **every** modulation decision, the AMC server validates it by requesting a full PHY-layer simulation from Sionna:

### Simplified View

```mermaid
sequenceDiagram
    participant NS3 as NS-3 / Client
    participant AMC as AMC Server :9001
    participant SIO as Sionna Server :9000
    participant MS as Metrics Server :5001

    NS3->>AMC: get_modulation {snr, flow_id, channel_info}

    Note over AMC: GAN predicts SNR → Table/RL picks modulation

    AMC->>SIO: {modulation, snr_db, payload, channel_params}
    SIO-->>AMC: {ber, bler, throughput, success}

    AMC->>MS: HTTP POST /ingest_event (async)

    AMC-->>NS3: {modulation, transmission_result, latencies}
```

### Detailed View

```mermaid
sequenceDiagram
    participant NS3 as NS-3 / Client
    participant AMC as AMC Server :9001
    participant GAN as GAN Predictor
    participant RL as RL Agent / Table
    participant SIO as Sionna Server :9000
    participant EP as Event Publisher
    participant MS as Metrics Server :5001

    NS3->>AMC: get_modulation {snr, flow_id, channel_info}

    Note over AMC: Check for preselected<br/>GAN prediction

    alt GAN prediction valid
        AMC->>AMC: Use preselected modulation
    else Prediction invalid or GAN disabled
        alt Table method
            AMC->>RL: SNR → lookup table
            RL-->>AMC: modulation (e.g. qam64)
        else RL method
            AMC->>RL: encode_state → select_action
            RL-->>AMC: modulation (e.g. qam64)
        end
    end

    Note over AMC: Proactive prediction<br/>for NEXT packet (if GAN enabled)
    AMC->>GAN: predict_next_snr(history, aux)
    GAN-->>AMC: predicted future SNR
    AMC->>RL: Decide modulation for predicted SNR
    AMC->>AMC: Store preselection for next request

    AMC->>SIO: {modulation, snr_db, 1024 random bits,<br/>channel_type, channel_params}
    SIO->>SIO: Sionna PHY simulation<br/>(QAM mapping → channel → AWGN → demapping)
    SIO-->>AMC: {ber, bler, effective_throughput,<br/>success, processing_time_ms}

    AMC->>EP: Publish event (async queue)
    EP->>MS: HTTP POST /ingest_event

    AMC-->>NS3: {modulation, method, transmission_result,<br/>decision_info, performance}
```

**Sionna request payload:**
```json
{
  "type": "amc_server",
  "id": 1,
  "k": 1024,
  "modulation": "qam64",
  "snr_db": 18.5,
  "payload": [0, 1, 1, 0, ...],
  "channel_type": "rayleigh",
  "channel_params": {
    "carrier_frequency": 2.4e9,
    "speed": 3.0,
    "distance": 100.0,
    "num_paths": 6,
    "delay_spread": 1e-6,
    "k_factor": 10.0,
    "phase_noise_std": 0.01
  }
}
```

**Sionna response:**
```json
{
  "id": 1,
  "success": true,
  "ber": 0.000234,
  "bler": 0.031,
  "effective_throughput": 4.82,
  "modulation": "qam64",
  "channel_type": "rayleigh",
  "processing_time_ms": 28.4,
  "realistic": true
}
```

---

### ④ AMC Server → NS-3 / Client (TCP response)

The full response back to the requester:

```json
{
  "modulation": "qam64",
  "method": "integrated_gan_amc_preselected_proactive_rl_with_gan",
  "flow_id": 1,
  "decision_info": {
    "method": "preselected_proactive_rl_with_gan",
    "prediction_info": {
      "used_preselected": true,
      "predicted_snr": 19.2,
      "actual_snr": 18.5,
      "prediction_error": 0.7
    }
  },
  "transmission_result": {
    "ber": 0.000234,
    "bler": 0.031,
    "throughput": 4.82,
    "success": true
  },
  "performance": {
    "decision_latency_ms": 42.3,
    "sionna_latency_ms": 28.4,
    "gan_latency_ms": 6.1
  }
}
```

---

### ⑤ ⑥ AMC → Event Publisher → Metrics Server (HTTP :5001)

Every decision is asynchronously published via `EventPublisher`:

```mermaid
flowchart LR
    AMC["AMC Decision"] -->|"event dict"| Q["Thread-safe Queue<br/>(max 10,000)"]
    Q -->|"background thread"| DISK["JSONL File<br/>(optional)"]
    Q -->|"HTTP POST<br/>/ingest_event"| MS["Metrics Server<br/>:5001"]
    MS -->|"deque buffer<br/>(max 5,000)"| MEM["In-Memory<br/>Event Store"]
```

**Event payload:**
```json
{
  "run_id": "run-default",
  "model_mode": "integrated_gan_rl",
  "model_name": "Integrated AMC (GAN+RL)",
  "timestamp": 1716579957.42,
  "flow_id": 1,
  "snr": 18.5,
  "chosen_modulation": "qam64",
  "decision_method": "preselected_proactive_rl_with_gan",
  "prediction_error": 0.7,
  "decision_latency_ms": 42.3,
  "sionna_latency_ms": 28.4,
  "gan_latency_ms": 6.1,
  "ber": 0.000234,
  "bler": 0.031,
  "throughput": 4.82,
  "success": true
}
```

---

### ⑦ ⑧ ⑨ Consumers ← Metrics Server (HTTP GET)

| Consumer | Endpoint | Purpose |
|---|---|---|
| **Dashboard** (:5000) | `GET /api/events?window=50` | Real-time charts (Chart.js) |
| **Benchmark Client** | `GET /api/events?window=N` | Time-series snapshots & plots |
| **Plot Figures** | `GET /api/events` | Post-hoc benchmark analysis |

The Metrics Server also exposes `GET /health` for status checks.

---

## Port & Protocol Summary

| Component | Port | Protocol | Role |
|---|---|---|---|
| **Sionna Server** | `9000` | Raw TCP + JSON | PHY-layer simulation (TensorFlow/Sionna) |
| **AMC Server** | `9001` | Raw TCP + JSON | AMC decision engine (GAN + RL/Table) |
| **Metrics Server** | `5001` | HTTP REST (Flask) | Event ingestion & distribution |
| **Dashboard** | `5000` | HTTP (Flask + HTML) | Real-time visualization |

---

## Internal AMC Decision Logic

```mermaid
flowchart TD
    REQ["Incoming Request<br/>{snr, flow_id, channel_info}"] --> CHECK{"Preselected<br/>GAN prediction<br/>available?"}

    CHECK -->|Yes| VALID{"Prediction still valid?<br/>|error| ≤ 3 dB<br/>AND modulation safe?"}
    VALID -->|Yes| USE_PRE["Use preselected<br/>modulation<br/><i>proactive decision</i>"]
    VALID -->|No| REACT["Reactive fallback"]

    CHECK -->|"No (or GAN disabled)"| REACT

    REACT --> METHOD{"Decision method?"}
    METHOD -->|Table| TABLE["3GPP Lookup Table<br/>SNR thresholds:<br/>≥15.5 → QAM256<br/>≥8.5 → QAM64<br/>≥1.5 → QAM16<br/>else → QAM4"]
    METHOD -->|RL| RLAGENT["RL Agent<br/>DQN with safety rules<br/>encode_state → select_action"]

    USE_PRE --> SIONNA_SIM["Validate via Sionna<br/>TCP :9000"]
    TABLE --> SIONNA_SIM
    RLAGENT --> SIONNA_SIM

    SIONNA_SIM --> PUBLISH["Publish event<br/>→ Metrics Server"]
    SIONNA_SIM --> RESPOND["Respond to client"]

    REQ --> GAN_NEXT["GAN: Predict NEXT SNR<br/>(if enabled)"]
    GAN_NEXT --> PRESELECT["Preselect modulation<br/>for next packet"]

    style REQ fill:#2d3748,color:#fff
    style USE_PRE fill:#805ad5,color:#fff
    style TABLE fill:#38a169,color:#fff
    style RLAGENT fill:#2b6cb0,color:#fff
    style SIONNA_SIM fill:#c53030,color:#fff
    style GAN_NEXT fill:#805ad5,color:#fff
```
