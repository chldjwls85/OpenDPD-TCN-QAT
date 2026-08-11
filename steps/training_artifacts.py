"""Atomic publication helpers for checkpoints produced by one training run."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_checkpoint(proj, requested: str, label: str) -> Path:
    """Publish only a checkpoint newly saved by this training invocation."""

    source = Path(proj.path_save_file_best).expanduser().resolve()
    if getattr(proj.logger, "checkpoint_save_count", 0) < 1:
        raise RuntimeError(
            f"{label} training did not save a new best checkpoint in this invocation"
        )
    if not source.is_file():
        raise FileNotFoundError(f"trained {label} checkpoint does not exist: {source}")

    if not requested:
        return source
    destination = Path(requested).expanduser().resolve()
    atomic_copy(source, destination)
    return destination
