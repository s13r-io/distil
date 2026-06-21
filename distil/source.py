"""Source metadata helpers for uploaded/pasted transcripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
_SEPARATORS = re.compile(r"[_\-.]+")
_SPACES = re.compile(r"\s+")
_KNOWN_SUFFIXES = {".srt", ".txt", ".md", ".markdown", ".text", ".vtt"}
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


class SourceUrlError(ValueError):
    """Raised when a provided source URL is not a supported YouTube URL."""


class SourceMetadataError(ValueError):
    """Raised when URL metadata cannot be fetched or parsed."""


@dataclass(frozen=True)
class SourceMetadata:
    title: str | None = None
    channel: str | None = None
    channel_url: str | None = None
    thumbnail_url: str | None = None
    metadata_provider: str | None = None
    metadata_fetched_at: str | None = None


def clean_source_title(raw: str | None, *, fallback: str = "Pasted transcript") -> str:
    """Turn an uploaded filename into a readable fallback title."""
    title = (raw or "").strip()
    if not title:
        return fallback
    title = Path(title).name
    suffix = Path(title).suffix.lower()
    if suffix in _KNOWN_SUFFIXES:
        title = title[: -len(suffix)]
    title = _BRACKETED.sub(" ", title)
    title = _SEPARATORS.sub(" ", title)
    title = _SPACES.sub(" ", title).strip(" -_.,")
    return title or fallback


def normalize_youtube_url(raw: str | None) -> str | None:
    """Validate and normalize an optional YouTube URL for attribution/navigation."""
    url = (raw or "").strip()
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or host not in _YOUTUBE_HOSTS:
        raise SourceUrlError("Source URL must be a YouTube URL.")
    return url


def fetch_youtube_oembed_metadata(url: str, *, timeout: float = 5.0) -> SourceMetadata:
    """Fetch public YouTube oEmbed metadata without requiring an API key."""
    normalized = normalize_youtube_url(url)
    if normalized is None:
        return SourceMetadata()
    endpoint = "https://www.youtube.com/oembed?format=json&url=" + quote(normalized, safe="")
    try:
        with urlopen(endpoint, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
    except Exception as exc:
        raise SourceMetadataError("Could not fetch YouTube metadata.") from exc
    if not isinstance(data, dict):
        raise SourceMetadataError("YouTube metadata response was not an object.")
    return parse_youtube_oembed_metadata(data)


def parse_youtube_oembed_metadata(data: dict) -> SourceMetadata:
    """Convert a YouTube oEmbed response into Distil source metadata."""
    return SourceMetadata(
        title=_clean_optional(data.get("title")),
        channel=_clean_optional(data.get("author_name")),
        channel_url=_clean_optional(data.get("author_url")),
        thumbnail_url=_clean_optional(data.get("thumbnail_url")),
        metadata_provider="youtube_oembed",
        metadata_fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def display_title(source_title: str, note_title: str | None = None) -> str:
    """Prefer the note title; otherwise show a cleaned source title."""
    note = (note_title or "").strip()
    if note:
        return note
    return clean_source_title(source_title, fallback="Untitled transcript")


def _clean_optional(value) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
