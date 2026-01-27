#!/bin/bash
# kd2 Multi-Agent Development Environment
# Usage: ./scripts/start-dev.sh [agent...]
#
# No args  → opens 4 windows: Architect, Tester, Dev, Porter
# With args → opens only specified agents
#   ./scripts/start-dev.sh architect dev
#   ./scripts/start-dev.sh mentor_dev

SESSION="kd2-dev"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Available agents (name must match .claude/agents/*.md frontmatter)
declare -A AGENT_LABELS=(
  [architect]="Architect"
  [tester]="Tester"
  [dev]="Dev"
  [porter]="Porter"
  [teacher]="Teacher"
  [researcher]="Researcher"
  [mentor_dev]="MentorDev"
)

# Default set if no args
if [ $# -eq 0 ]; then
  agents=(architect tester dev porter)
else
  agents=("$@")
fi

# Validate agent names
for agent in "${agents[@]}"; do
  if [ -z "${AGENT_LABELS[$agent]}" ]; then
    echo "Unknown agent: $agent"
    echo "Available: ${!AGENT_LABELS[*]}"
    exit 1
  fi
done

# Kill existing session if running
tmux kill-session -t "$SESSION" 2>/dev/null

# Create session with first agent
first="${agents[0]}"
tmux new-session -d -s "$SESSION" -n "${AGENT_LABELS[$first]}" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION" "claude --agent $first" C-m

# Create additional windows for remaining agents
for agent in "${agents[@]:1}"; do
  label="${AGENT_LABELS[$agent]}"
  tmux new-window -t "$SESSION" -n "$label" -c "$PROJECT_DIR"
  tmux send-keys -t "$SESSION" "claude --agent $agent" C-m
done

# Select first window
tmux select-window -t "$SESSION:0"

# Attach
tmux attach-session -t "$SESSION"
