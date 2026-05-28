#!/usr/bin/env bash

SESSION="amc-benchmarks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$(cd "$SCRIPT_DIR/../.." && pwd)/.env"
BENCHMARK_ARGS="$*"

START_AMC="set -a; source \"$ENV_FILE\"; set +a; source \"\$VENVNS3/bin/activate\"; python src/amc/amc.py"
START_SIONNA="set -a; source \"$ENV_FILE\"; set +a; source \"\$VENVNS3/bin/activate\"; python src/servers/sionna_server_torch.py"
START_NS3="set -a; source \"$ENV_FILE\"; set +a; source \"\$VENVNS3/bin/activate\"; make ns3-run"
START_BENCHMARKS="set -a; source \"$ENV_FILE\"; set +a; source \"\$VENVNS3/bin/activate\"; python src/runners/run_benchmarks.py $BENCHMARK_ARGS"

if tmux has-session -t "$SESSION" 2>/dev/null; then
	echo "tmux session '$SESSION' already exists; attaching without recreating panes."
else
	# Create a new detached session
	PANE_AMC="$(tmux new-session -d -s "$SESSION" -P -F '#{pane_id}')"

	# Pane 0: AMC Server (top-left)
	tmux send-keys -t "$PANE_AMC" "$START_AMC" C-m

	# Pane 1: Sionna Server (top-right)
	PANE_SIONNA="$(tmux split-window -h -t "$PANE_AMC" -P -F '#{pane_id}')"
	tmux send-keys -t "$PANE_SIONNA" "$START_SIONNA" C-m

	# Pane 2: ns-3 Docker (bottom-left)
	PANE_NS3="$(tmux split-window -v -t "$PANE_AMC" -P -F '#{pane_id}')"
	tmux send-keys -t "$PANE_NS3" "$START_NS3" C-m

	# Pane 3: Benchmarks (bottom-right)
	PANE_BENCHMARKS="$(tmux split-window -v -t "$PANE_SIONNA" -P -F '#{pane_id}')"
	tmux send-keys -t "$PANE_BENCHMARKS" "$START_BENCHMARKS" C-m
fi

# Attach to the session
tmux attach-session -t "$SESSION"