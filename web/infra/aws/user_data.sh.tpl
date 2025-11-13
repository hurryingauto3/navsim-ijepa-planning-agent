#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y docker docker-compose-plugin git curl

# Ensure docker-compose CLI is available (some environments lack the plugin)
curl -SL "https://github.com/docker/compose/releases/download/v2.30.3/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

mkdir -p /home/ec2-user/${project_root}
chown ec2-user:ec2-user /home/ec2-user/${project_root}

cat <<'EOF' >/etc/profile.d/navsim.sh
export NAVSIM_ROOT="/home/ec2-user/${project_root}"
EOF

