FROM continuumio/miniconda3:latest

# copy project files and requirements 
COPY README.md requirements.txt ./
COPY MCU_benchmark MCU_benchmark
COPY green_agent/src src

# Create conda env as root, install packages, then clean caches
RUN conda create -y -n mcu-agent python=3.11 pip \
 && conda install -y -n mcu-agent -c conda-forge openjdk=8 \
 && conda run -n mcu-agent pip install --no-cache-dir -r requirements.txt \
 && conda clean -afy

# Create non-root user and set ownership of home and conda env
RUN adduser --disabled-password --gecos "" agent \
 && mkdir -p /home/agent \
 && chown -R agent:agent /home/agent \
 && chown -R agent:agent /opt/conda/envs/mcu-agent

USER agent
WORKDIR /home/agent

ENTRYPOINT ["conda", "run", "-n", "mcu-agent", "--no-capture-output", "python", "-u", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9019
