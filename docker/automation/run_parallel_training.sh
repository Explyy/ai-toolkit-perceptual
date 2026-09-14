#!/bin/sh
set -eu

cd /app/ai-toolkit
exec python -m training_automation parallel-run
