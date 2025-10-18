#!/usr/bin/env bash
# Lightweight, idempotent installer to place Miniconda in /scratch/$USER
# - installs Miniconda3 into /scratch/$USER/miniconda3
# - configures conda to put envs and pkgs under /scratch/$USER
# - creates the 'navsim' environment from environment.yml (optional)

set -euo pipefail

USER_SCRATCH="/scratch/${USER}"
MINICONDA_DIR="${USER_SCRATCH}/miniconda3"
ENV_NAME="navsim"
NAVSIM_DIR_DEFAULT="${USER_SCRATCH}/navsim_workspace/navsim"
ENV_FILE="environment.yml"

show_help(){
  cat <<EOF
Usage: $(basename "$0") [--navsim-dir PATH] [--recreate] [--skip-env]

Installs Miniconda into ${MINICONDA_DIR} and (optionally) creates the
'${ENV_NAME}' conda environment from environment.yml located in the navsim
directory (default: ${NAVSIM_DIR_DEFAULT}).

Options:
  --navsim-dir PATH   Path to the navsim repo (default: ${NAVSIM_DIR_DEFAULT})
  --recreate          If conda env exists, remove and recreate it
  --skip-env          Install conda only; do not create the 'navsim' env
  -h, --help          Show this help and exit
EOF
}

NAVSIM_DIR="${NAVSIM_DIR_DEFAULT}"
RECREATE_ENV=false
SKIP_ENV=false

while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --navsim-dir) NAVSIM_DIR="$2"; shift 2;;
    --recreate) RECREATE_ENV=true; shift;;
    --skip-env) SKIP_ENV=true; shift;;
    -h|--help) show_help; exit 0;;
    *) echo "Unknown arg: $1"; show_help; exit 1;;
  esac
done

echo "Installing Miniconda into: ${MINICONDA_DIR}"
echo "Navsim dir: ${NAVSIM_DIR}"

mkdir -p "${USER_SCRATCH}"

if [[ -d "${MINICONDA_DIR}" ]]; then
  echo "Miniconda already present at ${MINICONDA_DIR}" 
  echo "If you want to reinstall, remove the directory and re-run this script."
else
  echo "Downloading Miniconda installer..."
  TMP=$(mktemp -d)
  INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  curl -L -o "${TMP}/${INSTALLER}" "https://repo.anaconda.com/miniconda/${INSTALLER}"
  chmod +x "${TMP}/${INSTALLER}"
  echo "Running Miniconda installer (silent)..."
  bash "${TMP}/${INSTALLER}" -b -p "${MINICONDA_DIR}"
  rm -rf "${TMP}"
  echo "Miniconda installed."
fi

CONDA_BIN="${MINICONDA_DIR}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "ERROR: conda binary not found at ${CONDA_BIN}" >&2
  exit 2
fi

echo "Configuring conda to use scratch paths for envs and pkgs..."
"${CONDA_BIN}" config --set envs_dirs "${USER_SCRATCH}/conda_envs" || true
"${CONDA_BIN}" config --set pkgs_dirs "${USER_SCRATCH}/conda_pkgs" || true

echo "Ensuring envs/pkgs dirs exist..."
mkdir -p "${USER_SCRATCH}/conda_envs" "${USER_SCRATCH}/conda_pkgs"

echo "Adding conda to PATH for this session..."
export PATH="${MINICONDA_DIR}/bin:${PATH}"

echo "Conda version: $(${CONDA_BIN} --version)"

if [[ "${SKIP_ENV}" == "true" ]]; then
  echo "Skipping environment creation as requested."
  echo "To create the navsim env later run:"
  echo "  ${CONDA_BIN} env create -n ${ENV_NAME} -f ${NAVSIM_DIR}/${ENV_FILE}"
  exit 0
fi

if [[ ! -d "${NAVSIM_DIR}" ]]; then
  echo "ERROR: navsim directory not found: ${NAVSIM_DIR}" >&2
  exit 3
fi

if [[ ! -f "${NAVSIM_DIR}/${ENV_FILE}" ]]; then
  echo "ERROR: environment.yml not found at ${NAVSIM_DIR}/${ENV_FILE}" >&2
  exit 4
fi

echo "Creating conda environment '${ENV_NAME}' from ${NAVSIM_DIR}/${ENV_FILE}"

if ${CONDA_BIN} env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  if [[ "${RECREATE_ENV}" == "true" ]]; then
    echo "Recreating existing environment: removing ${ENV_NAME}"
    ${CONDA_BIN} env remove -n ${ENV_NAME} -y
  else
    echo "Environment ${ENV_NAME} already exists. Use --recreate to recreate it."
    exit 0
  fi
fi

${CONDA_BIN} env create -n ${ENV_NAME} -f "${NAVSIM_DIR}/${ENV_FILE}"

echo "Environment '${ENV_NAME}' created."

cat <<EOF

Next steps:
  1. Add the following line to your shell init (e.g. ~/.bashrc):
       export PATH=\"${MINICONDA_DIR}/bin:\\$PATH\"
     Or run it in your session to use conda now.

  2. Activate the environment:
       conda activate ${ENV_NAME}

  3. Verify:
       conda --version
       conda env list

Notes:
  - Environments will be placed in ${USER_SCRATCH}/conda_envs
  - Conda packages will be cached in ${USER_SCRATCH}/conda_pkgs
  - This avoids touching system locations and keeps storage under /scratch

EOF

exit 0
