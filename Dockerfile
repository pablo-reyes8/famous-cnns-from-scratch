# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY . .
RUN python -m pip wheel --no-deps --wheel-dir /wheels .


FROM python:3.11-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=builder /wheels /wheels
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        "torch>=2.8.0,<3.0.0" "torchvision>=0.23.0,<0.24.0" \
    && python -m pip install /wheels/*.whl \
    && rm -rf /wheels

WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/outputs && chown -R app:app /workspace
USER app

ENTRYPOINT ["famous-cnns"]
CMD ["list"]
