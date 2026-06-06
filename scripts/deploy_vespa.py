"""
Deploy the Vespa application package (``vespa/``) to the Vespa config server.

Run automatically by the one-shot ``vespa-deploy`` compose service once Vespa is
healthy, so a single ``docker compose up`` brings Vespa up **and** activates the
schema — no manual ``curl`` step on the EC2 box. Idempotent: re-running just
prepares+activates a new session with the same package (safe to run on every
boot).

Uses only the standard library (no external deps) so it runs in any image.

Env:
    VESPA_CONFIG_ENDPOINT   config server base URL (default http://vespa:19071)
    VESPA_QUERY_ENDPOINT    query endpoint to wait for (default http://vespa:8080)
    VESPA_APP_DIR           path to the app package (default <repo>/vespa)
    VESPA_DEPLOY_TIMEOUT    seconds to wait for the config server (default 180)
"""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CONFIG = os.getenv("VESPA_CONFIG_ENDPOINT", "http://vespa:19071").rstrip("/")
QUERY = os.getenv("VESPA_QUERY_ENDPOINT", "http://vespa:8080").rstrip("/")
APP_DIR = Path(
    os.getenv("VESPA_APP_DIR", str(Path(__file__).resolve().parents[1] / "vespa"))
)
WAIT = int(os.getenv("VESPA_DEPLOY_TIMEOUT", "180"))


def _health_up(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/state/v1/health", timeout=5) as r:
            return r.status == 200 and '"up"' in r.read().decode()
    except Exception:
        return False


def _wait_healthy(base: str, label: str, deadline: float) -> bool:
    while time.time() < deadline:
        if _health_up(base):
            print(f"{label} is up ({base}).", flush=True)
            return True
        time.sleep(3)
    return False


def _zip_app(app_dir: Path) -> bytes:
    """Zip the application package (services.xml, hosts.xml, schemas/*.sd)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("services.xml", "hosts.xml"):
            p = app_dir / name
            if p.exists():
                z.write(p, name)
        schemas = app_dir / "schemas"
        for sd in sorted(schemas.glob("*.sd")):
            z.write(sd, f"schemas/{sd.name}")
    return buf.getvalue()


def _deploy(zip_bytes: bytes) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{CONFIG}/application/v2/tenant/default/prepareandactivate",
        data=zip_bytes,
        headers={"Content-Type": "application/zip"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> int:
    deadline = time.time() + WAIT
    print(f"Waiting up to {WAIT}s for Vespa config server at {CONFIG} ...", flush=True)
    if not _wait_healthy(CONFIG, "Config server", deadline):
        print("ERROR: Vespa config server never became healthy.", file=sys.stderr)
        return 1

    if not APP_DIR.exists():
        print(f"ERROR: app package dir not found: {APP_DIR}", file=sys.stderr)
        return 1

    print(f"Zipping application package from {APP_DIR} ...", flush=True)
    zip_bytes = _zip_app(APP_DIR)

    print(f"Deploying ({len(zip_bytes)} bytes) to {CONFIG} ...", flush=True)
    status, body = _deploy(zip_bytes)
    if status != 200:
        print(f"ERROR: deploy failed (HTTP {status}): {body[:600]}", file=sys.stderr)
        return 1

    print("Vespa application deployed and activated.", flush=True)
    # Best-effort: wait for the query endpoint to start serving before we exit,
    # so dependents that gate on this service see a queryable Vespa.
    _wait_healthy(QUERY, "Query endpoint", time.time() + 120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
