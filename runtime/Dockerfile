# syntax=docker/dockerfile:1
FROM rust:1.87-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/runtime /usr/local/bin/runtime
ENTRYPOINT ["/usr/local/bin/runtime"]
