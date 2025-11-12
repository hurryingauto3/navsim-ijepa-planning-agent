import os
import pathlib

import boto3


BUCKET = os.getenv("R2_BUCKET")
ENDPOINT = os.getenv("R2_ENDPOINT")
ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

if not all([BUCKET, ENDPOINT, ACCESS_KEY, SECRET_KEY]):
    raise SystemExit("Missing required R2 environment variables")

client = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

CACHE_DIR = pathlib.Path("data/cached_runs")
for path in CACHE_DIR.glob("*.json"):
    key = f"cached_runs/{path.name}"
    client.upload_file(path.as_posix(), BUCKET, key)
    print("uploaded", path.name)
