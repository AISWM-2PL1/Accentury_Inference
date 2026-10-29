FROM 325771561913.dkr.ecr.ap-northeast-2.amazonaws.com/accentury/inference-releases@sha256:f360ac06f2c7374fc18b5af8c1ce860ba0c668b5d01aab92bec3ff8605884935 AS pruned
USER root
ENV ACCENTURY_SLIM_TEST=1
COPY code/prune_image.py code/vendor_transformers.py /tmp/
RUN printf '%s' isolated-kan296-slimming-2026-10-29 > /tmp/accentury-slim-build-guard && python -B /tmp/prune_image.py && python -B /tmp/vendor_transformers.py && rm /tmp/prune_image.py /tmp/vendor_transformers.py /tmp/accentury-slim-build-guard
FROM scratch
COPY --from=pruned / /
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG="C.UTF-8"
ENV GPG_KEY="7169605F62C751356D054A26A821E680E5FA6305"
ENV PYTHON_VERSION="3.12.14"
ENV PYTHON_SHA256="5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a"
ENV DEBIAN_FRONTEND="noninteractive"
ENV PIP_NO_CACHE_DIR="1"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV ACCENTURY_MFA_BIN="/opt/conda/envs/mfa/bin"
ENV ACCENTURY_MFA_DICT="/app/dict_aug_track1.txt"
ENV ACCENTURY_DATA="/data"
ENV HF_HOME="/opt/hf"
ENV HF_HUB_OFFLINE="1"
ENV MFA_ROOT_DIR="/opt/mfa"
ENV ACCENTURY_CODE_HASH="0d72c03"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"
WORKDIR /app
USER accentury
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
