## AWS One-Button Hosting

This folder holds minimal IaC + Docker assets to deploy the web showcase on a single EC2 instance (Amazon Linux 2023) and wire it to a subdomain such as `demo.alihamzas.com`.

### What gets created
- Security group exposing `80/443` to the internet and `22` to a configurable CIDR.
- IAM role/instance profile with SSM access (so you can troubleshoot via AWS Console if SSH is locked down).
- EC2 instance (default `t3.micro`, 40 GB gp3 root disk) with Docker + Compose pre-installed through cloud-init. You can override `var.instance_type` if you need more CPU, but `t3.micro` is Free Tier eligible.
- Optional Route53 `A` record when you supply `route53_zone_id` + `subdomain` (e.g., `demo`).

### Manual one-off usage
```bash
cd web/infra/aws
terraform init
terraform apply \
  -var="instance_type=t3.micro" \
  -var="ssh_key_name=navsim-ci" \
  -var="allow_ssh_cidr=YOUR.IP.ADDR.0/32" \
  -var="route53_zone_id=Z123456789" \
  -var="subdomain=demo"

# After apply, copy the entire web folder and start compose:
rsync -az ../.. ec2-user@<PUBLIC_IP>:/home/ec2-user/navsim-web
ssh ec2-user@<PUBLIC_IP> <<'EOF'
cd /home/ec2-user/navsim-web/web/infra/aws
sudo docker compose -f docker-compose.aws.yml build --pull
sudo docker compose -f docker-compose.aws.yml up -d --remove-orphans
EOF
```

### GitHub Actions workflow
`.github/workflows/deploy.yml` automates the flow:
1. Trigger on pushes to anything inside `web/**`.
2. Runs Terraform (init/plan/apply) in this directory.
3. Uses an SSH key stored in `AWS_SSH_KEY` secret to `rsync` the `web/` directory and run the compose stack remotely.

Required repository secrets:
| Secret | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Terraform + rsync role with EC2, Route53, SSM permissions |
| `AWS_REGION` | matches `var.aws_region` (defaults to `us-east-1`) |
| `AWS_SSH_KEY` | private key that corresponds to EC2 `ssh_key_name` |
| `AWS_SSH_KEY_NAME` | exact key pair name Terraform should reference |
| `AWS_SSH_ALLOWED_CIDR` (optional) | overrides default `0.0.0.0/0` for port 22 |
| `AWS_INSTANCE_TYPE` (optional) | overrides the default `t3.micro` instance size |
| `ROUTE53_ZONE_ID` / `ROUTE53_SUBDOMAIN` (optional) | publish `demo.alihamzas.com` directly from Terraform |

Once deployed you can point `demo.alihamzas.com` (or any other subdomain under `alihamzas.com`) at the EC2 public IP if you manage DNS elsewhere.

