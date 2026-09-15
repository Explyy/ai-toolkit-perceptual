#!/bin/sh
set -eu

config_path="${TRAINING_UNIFIED_CONFIG:-/storage/config/klein-unified.yaml}"
gui_start="${TRAINING_GUI_START:-/start.sh}"
log_path="${TRAINING_UNIFIED_LOG:-/storage/automation/unified/supervisor.log}"

if [ ! -f "$config_path" ]; then
  echo "Unified automation config is missing: $config_path" >&2
  exit 64
fi
if [ ! -x "$gui_start" ]; then
  echo "Inherited GUI start executable is unavailable: $gui_start" >&2
  exit 64
fi

mkdir -p "$(dirname "$log_path")"
cd /app/ai-toolkit
python -m training_automation prepare-unified "$config_path" >>"$log_path" 2>&1
python -m training_automation.unified_supervisor "$config_path" >>"$log_path" 2>&1 &
echo "Unified automation started; durable status: ${TRAINING_UNIFIED_STATUS:-/storage/automation/unified/supervisor-state.json}"
exec "$gui_start"
