FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/CEMeNT-PSAAP/MCDC"
LABEL org.opencontainers.image.description="MC/DC: Monte Carlo Dynamic Code"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

# Disable apt sandbox for rootless container compatibility (HPC systems)
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/no-sandbox

# MPICH is used instead of OpenMPI for broader HPC compatibility:
# - No openssh-client dependency (avoids chmod issues on some filesystems)
# - No root restrictions (runs as root without --allow-run-as-root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libmpich-dev \
    mpich \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /opt/mcdc
COPY pyproject.toml .
RUN pip install --no-cache-dir mpi4py
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]