FROM python:3.12-slim-bookworm

# Minimal build tools (add more as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates build-essential pkg-config \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*


# Create non-root user for VS Code
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

WORKDIR /workspaces
USER ${USERNAME}
