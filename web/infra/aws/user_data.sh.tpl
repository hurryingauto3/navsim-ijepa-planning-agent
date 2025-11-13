#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y docker docker-compose-plugin git

systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

mkdir -p /home/ec2-user/${project_root}
chown ec2-user:ec2-user /home/ec2-user/${project_root}

cat <<'EOF' >/etc/profile.d/navsim.sh
export NAVSIM_ROOT="/home/ec2-user/${project_root}"
EOF

