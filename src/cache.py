import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "cache"
)

CACHE_FORMAT_VERSION = 1

ModelType = TypeVar(
    "ModelType",
    bound=BaseModel,
)


def resolve_project_path(
    file_path: str,
) -> Path:
    """Resolve an absolute or project-relative file path."""

    path = Path(
        file_path
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def file_sha256(
    file_path: str,
) -> str:
    """Calculate the SHA-256 hash of a file."""

    path = resolve_project_path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def optional_file_sha256(
    file_path: str,
) -> Optional[str]:
    """Return a file hash, or None when the file is absent."""

    path = resolve_project_path(
        file_path
    )

    if not path.exists():
        return None

    return file_sha256(
        str(path)
    )


def build_model_cache_key(
    *,
    tool_name: str,
    tool_version: str,
    image_sha256: str,
    inputs: Dict[str, Any],
) -> str:
    """Build a deterministic cache key."""

    key_payload = {
        "cache_format_version": (
            CACHE_FORMAT_VERSION
        ),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "image_sha256": image_sha256,
        "inputs": inputs,
    }

    canonical_payload = json.dumps(
        key_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_payload.encode(
            "utf-8"
        )
    ).hexdigest()


def sanitize_tool_name(
    tool_name: str,
) -> str:
    """Convert a tool name into a safe directory name."""

    safe_characters = []

    for character in tool_name:
        if (
            character.isalnum()
            or character in {"-", "_"}
        ):
            safe_characters.append(
                character
            )
        else:
            safe_characters.append(
                "_"
            )

    return "".join(
        safe_characters
    )


def get_cache_path(
    *,
    tool_name: str,
    cache_key: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """Return the JSON path for one cache entry."""

    root = (
        cache_root
        if cache_root is not None
        else DEFAULT_CACHE_ROOT
    )

    tool_directory = (
        root
        / sanitize_tool_name(
            tool_name
        )
    )

    return (
        tool_directory
        / f"{cache_key}.json"
    )


def load_cached_model(
    *,
    tool_name: str,
    cache_key: str,
    model_type: Type[ModelType],
    cache_root: Optional[Path] = None,
) -> Optional[ModelType]:
    """Load and validate a cached Pydantic model."""

    cache_path = get_cache_path(
        tool_name=tool_name,
        cache_key=cache_key,
        cache_root=cache_root,
    )

    if not cache_path.exists():
        return None

    try:
        raw_cache = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            raw_cache.get(
                "cache_format_version"
            )
            != CACHE_FORMAT_VERSION
        ):
            return None

        cached_value = raw_cache[
            "value"
        ]

        return model_type.model_validate(
            cached_value
        )

    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
    ):
        try:
            cache_path.unlink()
        except OSError:
            pass

        return None


def save_cached_model(
    *,
    tool_name: str,
    tool_version: str,
    cache_key: str,
    model: BaseModel,
    cache_root: Optional[Path] = None,
) -> Path:
    """Atomically save a Pydantic model as JSON."""

    cache_path = get_cache_path(
        tool_name=tool_name,
        cache_key=cache_key,
        cache_root=cache_root,
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_payload = {
        "cache_format_version": (
            CACHE_FORMAT_VERSION
        ),
        "tool_name": tool_name,
        "tool_version": tool_version,
        "model_type": (
            type(model).__name__
        ),
        "value": model.model_dump(
            mode="json"
        ),
    }

    temporary_path = (
        cache_path.parent
        / (
            f".{cache_path.name}."
            f"{uuid4().hex}.tmp"
        )
    )

    try:
        temporary_path.write_text(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            cache_path
        )

    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return cache_path