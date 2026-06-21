"""Source metadata helpers: title cleanup and YouTube URL validation."""

import pytest

from distil.source import (
    SourceUrlError,
    clean_source_title,
    normalize_youtube_url,
    parse_youtube_oembed_metadata,
)


@pytest.mark.unit
def test_clean_source_title_removes_extension_brackets_and_separators():
    title = clean_source_title("[English] my_video-title.final (Transcript).srt")
    assert title == "my video title final"


@pytest.mark.unit
def test_clean_source_title_falls_back_when_empty():
    assert clean_source_title("[English].txt", fallback="Pasted transcript") == "Pasted transcript"


@pytest.mark.unit
def test_normalize_youtube_url_accepts_youtube_hosts():
    assert normalize_youtube_url("youtube.com/watch?v=abc") == "https://youtube.com/watch?v=abc"
    assert normalize_youtube_url("https://youtu.be/abc") == "https://youtu.be/abc"


@pytest.mark.unit
def test_normalize_youtube_url_rejects_other_hosts():
    with pytest.raises(SourceUrlError):
        normalize_youtube_url("https://example.com/watch?v=abc")


@pytest.mark.unit
def test_parse_youtube_oembed_metadata_keeps_useful_fields():
    meta = parse_youtube_oembed_metadata({
        "title": "A Better Video Title",
        "author_name": "Useful Channel",
        "author_url": "https://www.youtube.com/@useful",
        "thumbnail_url": "https://i.ytimg.com/vi/abc/hqdefault.jpg",
    })
    assert meta.title == "A Better Video Title"
    assert meta.channel == "Useful Channel"
    assert meta.channel_url == "https://www.youtube.com/@useful"
    assert meta.thumbnail_url == "https://i.ytimg.com/vi/abc/hqdefault.jpg"
    assert meta.metadata_provider == "youtube_oembed"
    assert meta.metadata_fetched_at
