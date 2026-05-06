#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/AudioToolAgent"
nohup python src/tool_server.py --port 16181 > logs/tool_server.log 2>&1 &
echo "Tool server started on port 16181. Logging to logs/tool_server.log"
