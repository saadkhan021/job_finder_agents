from __future__ import annotations

import re


def parse_relative_age_hours(text: str) -> float | None:
    """
    Parse strings like "3 hours ago", "Just posted", "Today",
    "1 day ago", "2 weeks ago" into an approximate age in hours.

    Returns None if the text doesn't contain a recognizable time signal
    (caller should treat that as "unknown age" and decide how to handle it).
    """

    if not text:
        return None

    normalized = text.strip().lower()

    if any(keyword in normalized for keyword in ("just posted", "just now", "today", "active")):
        return 0.0

    match = re.search(r"(\d+)\s*minute", normalized)
    if match:
        return float(match.group(1)) / 60

    match = re.search(r"(\d+)\s*hour", normalized)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+)\s*day", normalized)
    if match:
        return float(match.group(1)) * 24

    match = re.search(r"(\d+)\s*week", normalized)
    if match:
        return float(match.group(1)) * 24 * 7

    match = re.search(r"(\d+)\s*month", normalized)
    if match:
        return float(match.group(1)) * 24 * 30

    return None
