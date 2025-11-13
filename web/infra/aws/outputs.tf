output "instance_id" {
  value       = aws_instance.navsim.id
  description = "EC2 instance ID hosting navsim"
}

output "public_ip" {
  value       = aws_instance.navsim.public_ip
  description = "Public IP for SSH and DNS"
}

output "public_dns" {
  value       = aws_instance.navsim.public_dns
  description = "Public DNS record AWS assigns"
}

output "security_group_id" {
  value       = aws_security_group.navsim.id
  description = "Security group controlling inbound traffic"
}

