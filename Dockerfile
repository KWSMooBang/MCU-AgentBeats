FROM eclipse-temurin:8-jdk-jammy AS jdk8

FROM ghcr.io/astral-sh/uv:python3.13-trixie

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=jdk8 $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY MCU_benchmark MCU_benchmark
COPY green_agent/src src


RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9019
