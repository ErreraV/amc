#!/usr/bin/env bash

SESSION="pidr-sim"

# Create a new detached session
tmux new-session -d -s $SESSION

# Pane 0: AMC Server
tmux send-keys -t $SESSION:0 'python src/amc/amc.py' C-m

# Pane 1: Sionna Server
tmux split-window -h -t $SESSION:0
tmux send-keys -t $SESSION:0.1 'python src/servers/sionna_serverTorch.py' C-m

# Pane 2: ns-3 Docker
tmux split-window -v -t $SESSION:0
tmux send-keys -t $SESSION:0.2 'sleep 3 && make ns3-deploy' C-m

# Attach to the session
tmux attach-session -t $SESSION