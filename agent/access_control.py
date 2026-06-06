"""
User access control for document filtering.
"""

import os
from typing import Optional


def assign_access_level(bucket_type: str) -> str:
    """Map an S3 bucket type to a document access level."""
    return "private" if bucket_type == "private" else "public"


def can_access_document(user_id: Optional[str], access_level: str) -> bool:
    """
    Return True if the user is allowed to read a document with the given access level.

    Rules:
    - public  → everyone
    - private → admin users OR users listed in PRIVATE_DOCUMENT_USERS
    """
    if access_level == "public":
        return True

    if not user_id:
        return False

    admin_users = {
        u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()
    }
    if user_id in admin_users:
        return True

    private_users = {
        u.strip()
        for u in os.getenv("PRIVATE_DOCUMENT_USERS", "").split(",")
        if u.strip()
    }
    return user_id in private_users
