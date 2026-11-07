#!/usr/bin/env python3
"""Fetch daily download and engagement stats for TonicFind and send via Resend email API."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

TARGET_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "boobalan_arjunan@composermind.com")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "TonicFind Stats <onboarding@resend.dev>")


def fetch_stats():
    total_with_mirrors, total_without_mirrors = 0, 0
    last_day, last_week = 0, 0
    try:
        req = urllib.request.Request(
            "https://pypistats.org/api/packages/tonicfind/overall",
            headers={"User-Agent": "TonicFind-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
            rows = d.get("data", [])
            total_with_mirrors = sum(r["downloads"] for r in rows if r["category"] == "with_mirrors")
            total_without_mirrors = sum(r["downloads"] for r in rows if r["category"] == "without_mirrors")
    except Exception as e:
        print(f"[WARN] PyPI overall stats failed: {e}")

    try:
        req = urllib.request.Request(
            "https://pypistats.org/api/packages/tonicfind/recent",
            headers={"User-Agent": "TonicFind-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
            data = d.get("data", {})
            last_day = data.get("last_day", 0)
            last_week = data.get("last_week", 0)
    except Exception as e:
        print(f"[WARN] PyPI recent stats failed: {e}")

    return {
        "total_without_mirrors": total_without_mirrors,
        "total_with_mirrors": total_with_mirrors,
        "last_day": last_day,
        "last_week": last_week
    }


def send_email(stats):
    if not RESEND_API_KEY:
        print("[INFO] No RESEND_API_KEY provided. Skipping email dispatch.")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    html_content = f"""
    <h2>TonicFind Daily Analytics Report — {today}</h2>
    <p>Daily performance metrics for <b>tonicfind</b>:</p>
    <ul>
      <li><b>Downloads Yesterday:</b> {stats['last_day']}</li>
      <li><b>Downloads Past 7 Days:</b> {stats['last_week']}</li>
      <li><b>Total Organic Downloads:</b> {stats['total_without_mirrors']}</li>
      <li><b>Total with Mirrors:</b> {stats['total_with_mirrors']}</li>
    </ul>
    <p><i>Automated report sent to {TARGET_EMAIL}</i></p>
    """

    payload = json.dumps({
        "from": FROM_EMAIL,
        "to": [TARGET_EMAIL],
        "subject": f"📊 TonicFind Daily Analytics ({today}): {stats['last_day']} downloads",
        "html": html_content
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "TonicFind-Reporter"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[SUCCESS] Email sent: HTTP {resp.status}")
    except Exception as e:
        print(f"[ERROR] Failed to send email via Resend: {e}", file=sys.stderr)


def main():
    stats = fetch_stats()
    print(f"Stats fetched: {stats}")
    send_email(stats)


if __name__ == "__main__":
    main()
