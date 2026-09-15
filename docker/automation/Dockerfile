FROM explyy/ai-toolkit-perceptual@sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09

USER root
WORKDIR /app/ai-toolkit

# Install this checkout's additive package, then patch only four verified
# trainer anchors. The base image's perceptual trainer and UI remain intact.
COPY training_automation/ /app/ai-toolkit/training_automation/
COPY config/examples/klein_automation/ /app/ai-toolkit/config/examples/klein_automation/
COPY docker/automation/install_overlay.py /tmp/install_training_automation_overlay.py
COPY docker/automation/install_evaluation_models.py /tmp/install_evaluation_models.py
COPY docker/automation/smoke_gpu_isolation.py /tmp/smoke_gpu_isolation.py
COPY docker/automation/run_parallel_training.sh /run-parallel-training
RUN mkdir -p /storage/automation /storage/config /storage/datasets /storage/output \
    && python /tmp/install_training_automation_overlay.py --root /app/ai-toolkit \
    && python /tmp/install_evaluation_models.py \
    && chmod 0755 /run-parallel-training \
    && python -m compileall -q /app/ai-toolkit/training_automation \
    && python -m compileall -q /app/ai-toolkit/jobs/process/BaseSDTrainProcess.py \
    && python -m training_automation.recipe /app/ai-toolkit/config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml /app/ai-toolkit \
    && python -m training_automation --help >/dev/null \
    && python -c "from training_automation.worker import worker_log_path; assert worker_log_path({'TRAINING_RUN_ID':'smoke','TRAINING_SHARD_ID':'a'}) == __import__('pathlib').Path('/storage/automation/smoke/a/worker.log')" \
    && python -m training_automation run /app/ai-toolkit/config/examples/klein_automation/automation.yaml --dry-run >/tmp/training-automation-dry-run.json \
    && python -c "import json; from training_automation.backends import InsightFaceCPUBackend; p='/opt/training-automation-models'; s=json.load(open(p + '/SOURCES.json')); assert len(s['archive_sha256']) == 64; InsightFaceCPUBackend(model_dir=p + '/insightface/models/buffalo_l')" \
    && python /tmp/smoke_gpu_isolation.py \
    && rm /tmp/install_training_automation_overlay.py /tmp/install_evaluation_models.py /tmp/smoke_gpu_isolation.py

ENV TRAINING_AUTOMATION_ROOT=/storage/automation

# CMD and entrypoint are inherited to preserve the image's UI launch behavior.
