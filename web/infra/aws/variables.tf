variable "aws_region" {
  description = "Region to deploy the navsim showcase"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance size"
  type        = string
  default     = "t3.micro"
}

variable "project_name" {
  description = "Prefix for AWS resources and remote install path"
  type        = string
  default     = "navsim-web"
}

variable "ssh_key_name" {
  description = "Existing AWS EC2 key pair name for SSH/rsync"
  type        = string
}

variable "allow_ssh_cidr" {
  description = "CIDR allowed to reach port 22"
  type        = string
  default     = "0.0.0.0/0"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 40
}

variable "route53_zone_id" {
  description = "Optional Route53 Hosted Zone ID for alihamzas.com"
  type        = string
  default     = ""
}

variable "subdomain" {
  description = "Optional subdomain record (e.g., demo.alihamzas.com)"
  type        = string
  default     = ""
}

