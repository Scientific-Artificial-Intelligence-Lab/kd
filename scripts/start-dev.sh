#!/bin/bash
# kd2 Multi-Agent Development Environment
# Usage: ./scripts/start-dev.sh
#
# Opens 4 tmux panes with agent sessions.
# Each pane starts claude with the corresponding agent prompt.

SESSION="kd2-dev"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Kill existing session if running
tmux kill-session -t "$SESSION" 2>/dev/null

# Create new session with Architect pane
tmux new-session -d -s "$SESSION" -n "agents" -c "$PROJECT_DIR"

# Pane 0: Architect (top-left)
tmux send-keys -t "$SESSION:0.0" \
  "echo '=== Architect ===' && claude --prompt 'You are the Architect agent. Read .claude/agents/architect.md for your role. 使用中文对话。'" C-m

# Pane 1: Tester (top-right)
tmux split-window -h -t "$SESSION:0.0" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.1" \
  "echo '=== Tester ===' && claude --prompt 'You are the Tester agent. Read .claude/agents/tester.md for your role. 使用中文对话。'" C-m

# Pane 2: Dev (bottom-left)
tmux split-window -v -t "$SESSION:0.0" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.2" \
  "echo '=== Dev ===' && claude --prompt 'You are the Dev agent. Read .claude/agents/dev.md for your role. 使用中文对话。'" C-m

# Pane 3: Mentor (bottom-right)
tmux split-window -v -t "$SESSION:0.1" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.3" \
  "echo '=== Mentor Dev ===' && claude --prompt 'You are the Mentor Dev agent. Read .claude/agents/mentor_dev.md for your role. 使用中文对话。'" C-m

# Select Architect pane
tmux select-pane -t "$SESSION:0.0"

# Attach
tmux attach-session -t "$SESSION"
