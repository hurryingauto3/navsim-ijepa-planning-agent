# Use an official NVIDIA CUDA image as the base
# Choose a version compatible with your host driver and environment.yml dependencies
# 'devel' includes compilers and headers, often useful for building packages
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# --- Install system dependencies (including Miniconda prerequisites) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    unzip \
    libgl1-mesa-glx \
    curl \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# --- Initialize Conda for bash ---
RUN conda init bash

# --- Set GDAL environment variables ---
# These might need adjustment depending on the base image's GDAL location if pre-installed,
# but installing libgdal-dev usually puts them in standard locations.
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# --- Set working directory ---
WORKDIR /navsim_workspace

# --- Clone the navsim repository and setup directories ---
# Using ARG to potentially allow overriding repo/branch during build
ARG NAVSIM_REPO=https://github.com/autonomousvision/navsim.git
ARG NAVSIM_BRANCH=main
RUN git clone --branch ${NAVSIM_BRANCH} ${NAVSIM_REPO} navsim && \
    chmod -R +x navsim/download

# --- Create conda environment using environment.yml from the cloned repository ---
# IMPORTANT: navsim/environment.yml MUST specify CUDA-enabled packages
# (e.g., pytorch compiled for CUDA from the correct channel)
RUN conda env create --name navsim --file navsim/environment.yml

# --- Activate the navsim environment for all subsequent commands ---
# Using SHELL to make subsequent RUN commands use bash with conda activated
SHELL ["/bin/bash", "-c"]
# Set path explicitly as well for ENV context and ENTRYPOINT
ENV PATH /opt/conda/envs/navsim/bin:$PATH

# --- Install navsim in editable mode and setup Jupyter with kernel ---
# Note: Activate is needed here as SHELL doesn't persist environment activation between RUNs in the same way
RUN source activate navsim && \
    pip install --no-cache-dir -e navsim && \
    conda install -y transformers jupyter ipykernel notebook ipywidgets && \
    python -m ipykernel install --user --name=navsim

# --- Install VS Code Server ---
# Ensure curl was installed above
RUN curl -fsSL https://code-server.dev/install.sh | sh

# --- Create config directory for code-server ---
RUN mkdir -p /root/.config/code-server

# --- Add default config with no authentication ---
RUN echo -e 'bind-addr: 0.0.0.0:8080\nauth: none\ncert: false' > /root/.config/code-server/config.yaml

# --- Set project-specific environment variables ---
ENV NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
    NUPLAN_MAPS_ROOT="/navsim_workspace/dataset/maps" \
    NAVSIM_EXP_ROOT="/navsim_workspace/exp" \
    NAVSIM_DEVKIT_ROOT="/navsim_workspace/navsim" \
    OPENSCENE_DATA_ROOT="/navsim_workspace/dataset" \
    HYDRA_FULL_ERROR=1 \
    JUPYTER_TOKEN="" \
    # Add NVIDIA specific env vars needed by some libraries
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all

# --- Expose the Jupyter and VS Code ports ---
EXPOSE 8888
EXPOSE 8080

# --- Copy and enable the entrypoint script ---
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Reset SHELL to default
SHELL ["/bin/sh", "-c"]