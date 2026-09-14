FROM explyy/ai-toolkit-perceptual@sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09

USER root
WORKDIR /app/ai-toolkit

# Install this checkout's additive package, then patch only four verified
# trainer anchors. The base image's perceptual trainer and UI remain intact.
COPY training_automation/ /app/ai-toolkit/training_automation/
COPY docker/automation/install_overlay.py /tmp/install_training_automation_overlay.py
RUN python /tmp/install_training_automation_overlay.py --root /app/ai-toolkit \
    && python -m compileall -q /app/ai-toolkit/training_automation \
    && mkdir -p /storage/automation /storage/config /storage/datasets /storage/output \
    && rm /tmp/install_training_automation_overlay.py

ENV TRAINING_AUTOMATION_ROOT=/storage/automation

# CMD and entrypoint are inherited to preserve the image's UI launch behavior.

