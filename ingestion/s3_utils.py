"""
AWS S3 utilities for listing and downloading documents.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PRIVATE_BUCKET = os.getenv("S3_PRIVATE_BUCKET", "")
_PUBLIC_BUCKET = os.getenv("S3_PUBLIC_BUCKET", "")

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}


def _get_bucket_name(bucket_type: str) -> str:
    if bucket_type == "private":
        return _PRIVATE_BUCKET
    return _PUBLIC_BUCKET


def _s3_client():
    import boto3

    # Supabase Storage exposes an S3-compatible API at /storage/v1/s3.
    # If SUPABASE_URL is set, route all S3 calls through it instead of AWS.
    supabase_url = os.getenv("SUPABASE_URL")
    endpoint_url = None
    if supabase_url:
        endpoint_url = f"{supabase_url.rstrip('/')}/storage/v1/s3"

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        endpoint_url=endpoint_url,
    )


def verify_s3_access() -> bool:
    """Return True if both configured S3 buckets are accessible."""
    try:
        client = _s3_client()
        for bucket in filter(None, [_PRIVATE_BUCKET, _PUBLIC_BUCKET]):
            client.head_bucket(Bucket=bucket)
        return True
    except Exception as exc:
        logger.error("S3 access check failed: %s", exc)
        return False


def list_documents_from_s3(bucket_type: str = "private", prefix: str = "") -> List[str]:
    """Return S3 object keys for supported document formats."""
    bucket = _get_bucket_name(bucket_type)
    if not bucket:
        logger.warning("No bucket configured for type '%s'", bucket_type)
        return []

    try:
        client = _s3_client()
        paginator = client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if Path(key).suffix.lower() in _SUPPORTED_EXTENSIONS:
                    keys.append(key)
        return keys
    except Exception as exc:
        logger.error("Failed to list documents from S3 bucket '%s': %s", bucket, exc)
        return []


def download_document_from_s3(
    doc_key: str,
    bucket_type: str = "private",
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Download a document from S3.

    If *save_path* is provided the content is also written to disk.
    Returns the raw bytes or None on failure.
    """
    bucket = _get_bucket_name(bucket_type)
    if not bucket:
        return None

    try:
        client = _s3_client()
        response = client.get_object(Bucket=bucket, Key=doc_key)
        content: bytes = response["Body"].read()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as fh:
                fh.write(content)

        return content
    except Exception as exc:
        logger.error("Failed to download '%s' from S3: %s", doc_key, exc)
        return None
