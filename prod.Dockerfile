# prod.Dockerfile
#  1. Line - Change builder image
FROM debian:buster-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends --yes python3-venv gcc libpython3-dev && \
    python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip

FROM builder AS builder-venv

COPY requirements.txt /requirements.txt
RUN /venv/bin/pip install -r /requirements.txt

FROM builder-venv AS tester

COPY . /app
WORKDIR /app
RUN /venv/bin/pytest

#  17. Line - Switch to Distroless image
FROM gcr.io/distroless/python3-debian10 AS runner

COPY --from=tester /venv /venv
COPY --from=tester /app /app

WORKDIR /app

ENTRYPOINT ["/venv/bin/python3", "-m", "kaggle_blueprint"]
USER 1001

LABEL name={NAME}
LABEL version={VERSION}
