# syntax=docker/dockerfile:1
#
# Builds AquaAlert (Rust) into ONE static binary and ships it on a distroless base.
# The binary has no runtime dependencies — no Python, no libpython, no glibc.
#
#   docker build -t aqua-alert-rs .
#   docker run --rm -p 8000:8000 aqua-alert-rs            # farmer screen + API on :8000
#   docker run --rm aqua-alert-rs demo                     # one-shot offline demo
#
# The ONNX deep-model path is NOT enabled here (no ~100 MB model baked into the image).
# Enable it by building with `--features onnx` and mounting a model + labels.json
# (see export/README.md); set AQUA_ONNX_MODEL / AQUA_ONNX_LABELS at runtime.

# ---------- build stage ----------
FROM rust:1.83-bookworm AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl
# link the musl target with musl-gcc and use it for any C shims
RUN mkdir -p /usr/local/cargo \
    && printf '[target.x86_64-unknown-linux-musl]\nlinker = "musl-gcc"\n' > /usr/local/cargo/config.toml
ENV CARGO_HOME=/usr/local/cargo
ENV CC_x86_64_unknown_linux_musl=musl-gcc

WORKDIR /app
# dependency layer first for better caching
COPY Cargo.toml ./
COPY src ./src
COPY assets ./assets
COPY export ./export
RUN cargo build --release --target x86_64-unknown-linux-musl --features server

# ---------- runtime stage ----------
FROM gcr.io/distroless/static-debian12
COPY --from=build /app/target/x86_64-unknown-linux-musl/release/aqua-alert /usr/local/bin/aqua-alert
EXPOSE 8000
ENV AQUA_ALERT_PORT=8000
ENTRYPOINT ["/usr/local/bin/aqua-alert"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
