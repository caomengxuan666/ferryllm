FROM rust:1-bookworm AS builder

WORKDIR /app
COPY . .

RUN cargo build --release --bin ferryllm

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/ferryllm /usr/local/bin/ferryllm
COPY examples/config /app/examples/config

EXPOSE 3000

ENTRYPOINT ["ferryllm"]
CMD ["serve", "--config", "examples/config/codexapis.toml"]
