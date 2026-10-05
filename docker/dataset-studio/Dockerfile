# Upstream 0.13.23 linux/amd64, verified registry manifest; no training overlay.
FROM ostris/aitoolkit@sha256:26cb1eb1d8aae154c1c7399004265782124d2f0c1b29cab9b6886345538ed8d4 AS toolkit
FROM node:22.22.3-bookworm-slim@sha256:16d364eebf6b62da439dc993d9b80940c78b0ca38438452f011ab9a25c752644 AS ui-build
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ openssl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app/ai-toolkit
COPY --from=toolkit /app/ai-toolkit/version.py ./version.py
COPY ui ./ui
COPY scripts/dataset_studio_cloud.py ./scripts/dataset_studio_cloud.py
COPY docker/dataset-studio ./docker/dataset-studio
COPY .github/workflows/dataset-studio.yml ./.github/workflows/dataset-studio.yml
RUN python3 scripts/dataset_studio_cloud.py manifest --toolkit /app/ai-toolkit > studio-source.json
WORKDIR /app/ai-toolkit/ui
# sqlite3 6.0.1's downloaded Linux prebuild needs GLIBC_2.38; bookworm has
# 2.36. Compile the locked package with this stage's existing Python/g++.
RUN npm ci && npm rebuild sqlite3 --build-from-source && node ../docker/dataset-studio/check_sqlite.cjs /app/ai-toolkit/ui && npx prisma generate && npm run test:studio && npm run build && npx tsc --noEmit --incremental false

FROM toolkit AS runtime
USER root
RUN apt-get update && apt-get install -y --no-install-recommends nginx openssl && rm -rf /var/lib/apt/lists/*
ARG STUDIO_RELEASE=unversioned
LABEL org.opencontainers.image.title="Dataset Studio" org.opencontainers.image.revision=$STUDIO_RELEASE
COPY --from=ui-build /usr/local/bin/node /usr/local/bin/node
COPY --from=ui-build /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && rm -rf /app/ai-toolkit/ui && rm -f /app/ai-toolkit/aitk_db.db /app/ai-toolkit/aitk_db.db-wal /app/ai-toolkit/aitk_db.db-shm
COPY --from=ui-build /app/ai-toolkit/ui /app/ai-toolkit/ui
COPY --from=ui-build /app/ai-toolkit/studio-source.json /app/ai-toolkit/studio-source.json
COPY scripts/dataset_studio_cloud.py /app/ai-toolkit/scripts/dataset_studio_cloud.py
COPY docker/dataset-studio/nginx.conf /etc/nginx/dataset-studio.conf
COPY docker/dataset-studio/check_runtime.py /tmp/check_studio_runtime.py
COPY docker/dataset-studio/check_sqlite.cjs /tmp/check_studio_sqlite.cjs
COPY docker/dataset-studio/test_gateway.py docker/dataset-studio/nginx.conf /tmp/studio-gateway/
RUN cd /app/ai-toolkit && /usr/local/bin/node /tmp/check_studio_sqlite.cjs /app/ai-toolkit/ui && python /tmp/check_studio_runtime.py && python /tmp/studio-gateway/test_gateway.py && rm -rf /tmp/check_studio_sqlite.cjs /tmp/check_studio_runtime.py /tmp/studio-gateway
WORKDIR /app/ai-toolkit
RUN STUDIO_RELEASE=$STUDIO_RELEASE python -c "import json,os,hashlib; from version import VERSION; json.dump({'studioRelease':os.environ['STUDIO_RELEASE'],'toolkitVersion':VERSION,'sourceManifestSha256':hashlib.sha256(open('studio-source.json','rb').read()).hexdigest(),'baseDigest':'sha256:26cb1eb1d8aae154c1c7399004265782124d2f0c1b29cab9b6886345538ed8d4'},open('studio-release.json','w'))"
ENV DATASET_STUDIO_MOUNT=/workspace DATASET_STUDIO_ROOT=/workspace/dataset-studio AI_TOOLKIT_DB_JOURNAL_MODE=DELETE NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
WORKDIR /app/ai-toolkit/ui
EXPOSE 8675
CMD ["python3", "/app/ai-toolkit/scripts/dataset_studio_cloud.py", "serve"]
