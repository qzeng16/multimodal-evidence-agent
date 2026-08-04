from pathlib import Path

from pydantic import BaseModel

from src.cache import (
    build_model_cache_key,
    file_sha256,
    get_cache_path,
    load_cached_model,
    save_cached_model,
)


class CachePayload(BaseModel):
    """Small model used to test cache serialization."""

    message: str
    score: float


def test_cache_key_is_deterministic() -> None:
    """Equivalent inputs should produce the same cache key."""

    first_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "targets": ["28th St."],
            "config": "abc",
        },
    )

    second_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "config": "abc",
            "targets": ["28th St."],
        },
    )

    assert first_key == second_key


def test_cache_key_changes_when_inputs_change() -> None:
    """Different tool inputs must not share a cache key."""

    first_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "targets": ["28th St."],
        },
    )

    second_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "targets": ["29th St."],
        },
    )

    assert first_key != second_key


def test_cache_key_changes_when_image_changes() -> None:
    """Different image hashes must produce different keys."""

    first_key = build_model_cache_key(
        tool_name="image_inspector",
        tool_version="v1",
        image_sha256="first-image",
        inputs={
            "claim": "Test claim",
        },
    )

    second_key = build_model_cache_key(
        tool_name="image_inspector",
        tool_version="v1",
        image_sha256="second-image",
        inputs={
            "claim": "Test claim",
        },
    )

    assert first_key != second_key


def test_cache_key_changes_when_tool_version_changes() -> None:
    """Changing a tool version should invalidate old cache entries."""

    first_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "targets": ["28th St."],
        },
    )

    second_key = build_model_cache_key(
        tool_name="ocr_tool",
        tool_version="v2",
        image_sha256="image-hash",
        inputs={
            "targets": ["28th St."],
        },
    )

    assert first_key != second_key


def test_save_and_load_cached_model(
    tmp_path: Path,
) -> None:
    """A cached Pydantic model should survive a round trip."""

    cache_key = build_model_cache_key(
        tool_name="cache_test",
        tool_version="v1",
        image_sha256="image-hash",
        inputs={
            "test": True,
        },
    )

    payload = CachePayload(
        message="cache works",
        score=0.99,
    )

    saved_path = save_cached_model(
        tool_name="cache_test",
        tool_version="v1",
        cache_key=cache_key,
        model=payload,
        cache_root=tmp_path,
    )

    loaded_payload = load_cached_model(
        tool_name="cache_test",
        cache_key=cache_key,
        model_type=CachePayload,
        cache_root=tmp_path,
    )

    assert saved_path.exists()
    assert loaded_payload == payload


def test_missing_cache_returns_none(
    tmp_path: Path,
) -> None:
    """A missing cache entry should be treated as a cache miss."""

    loaded_payload = load_cached_model(
        tool_name="cache_test",
        cache_key="missing-key",
        model_type=CachePayload,
        cache_root=tmp_path,
    )

    assert loaded_payload is None


def test_corrupted_cache_is_removed(
    tmp_path: Path,
) -> None:
    """Invalid JSON should be removed and treated as a miss."""

    cache_path = get_cache_path(
        tool_name="cache_test",
        cache_key="corrupted-key",
        cache_root=tmp_path,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    loaded_payload = load_cached_model(
        tool_name="cache_test",
        cache_key="corrupted-key",
        model_type=CachePayload,
        cache_root=tmp_path,
    )

    assert loaded_payload is None
    assert not cache_path.exists()


def test_invalid_cached_model_is_removed(
    tmp_path: Path,
) -> None:
    """A cache entry with the wrong schema should be removed."""

    cache_path = get_cache_path(
        tool_name="cache_test",
        cache_key="invalid-model",
        cache_root=tmp_path,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        """
{
  "cache_format_version": 1,
  "tool_name": "cache_test",
  "tool_version": "v1",
  "model_type": "CachePayload",
  "value": {
    "message": 123,
    "score": "not-a-number"
  }
}
""".strip(),
        encoding="utf-8",
    )

    loaded_payload = load_cached_model(
        tool_name="cache_test",
        cache_key="invalid-model",
        model_type=CachePayload,
        cache_root=tmp_path,
    )

    assert loaded_payload is None
    assert not cache_path.exists()


def test_file_sha256_depends_on_file_content(
    tmp_path: Path,
) -> None:
    """The file hash should change when file contents change."""

    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text(
        "first content",
        encoding="utf-8",
    )

    second_file.write_text(
        "second content",
        encoding="utf-8",
    )

    first_hash = file_sha256(
        str(first_file)
    )

    second_hash = file_sha256(
        str(second_file)
    )

    assert first_hash != second_hash
    assert len(first_hash) == 64
    assert len(second_hash) == 64


def test_same_file_content_has_same_sha256(
    tmp_path: Path,
) -> None:
    """Identical file contents should produce identical hashes."""

    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text(
        "same content",
        encoding="utf-8",
    )

    second_file.write_text(
        "same content",
        encoding="utf-8",
    )

    assert (
        file_sha256(str(first_file))
        == file_sha256(str(second_file))
    )
