# Tools (debug & test helpers)

This folder contains lightweight scripts used for local debugging and testing of the analytics pipeline. These are not required for production and are intended for developer use.

Available scripts:

- `test_analytics_flow.py`: Simulates AMC client requests and checks that the metrics server and dashboard are reachable. Useful to reproduce end-to-end flow problems.
  - Run from repository root with the package `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src/amc_/Project/src python src/amc_/Project/src/tools/test_analytics_flow.py --duration 60 --rate 5
```

- `test_dashboard_config.py`: Unit-style test that creates a `RealtimeAnalyzer`, injects it into the dashboard Flask app, and exercises the dashboard endpoints using Flask's test client.
  - Run from the repository root:

```bash
cd src/amc_/Project && PYTHONPATH=. python -m src.tools.test_dashboard_config
```

- `debug_analyzer.py`: Quick script to exercise analyzer creation, populate it with dummy metrics, set the analyzer into the dashboard Flask app, and call the health and stats endpoints via the test client.

- `send_modulation_requests.py`: Simple TCP sender that repeatedly issues `get_modulation` requests to an AMC server for load/testing.
  - Run it directly:

```bash
python src/amc_/Project/src/tools/send_modulation_requests.py --host 127.0.0.1 --port 9001 --duration 10 --rate 5
```

Notes:
- These scripts assume you run the development servers locally (metrics server on port `5001`, dashboard on `5000`, AMC on `9001`).
- If you change package layout, update the `PYTHONPATH`/module invocation accordingly.
