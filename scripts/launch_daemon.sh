#!/usr/bin/env bash

set -euo pipefail

# The hardware launch scripts need an explicit robot IP.
if [[ -z "${ROBOT_IP:-}" ]]; then
  echo "ROBOT_IP is not set"
  exit 1
fi

session="server"
if ! tmux has-session -t "${session}" 2>/dev/null; then
  tmux new-session -d -s "${session}"
  tmux split-window -v
fi

# Keep the original commands unchanged so operator workflows stay identical.
tmux send-keys -t "${session}.0" \
  "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=${ROBOT_IP}" ENTER

tmux send-keys -t "${session}.1" \
  "launch_gripper.py gripper=franka_hand gripper.cfg.robot_ip=${ROBOT_IP}" ENTER

tmux attach -t "${session}"
