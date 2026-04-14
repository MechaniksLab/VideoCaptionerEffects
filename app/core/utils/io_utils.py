from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    """Атомарная запись текста: сначала во временный файл, потом replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target.with_name(f".{target.name}.tmp")
    with open(tmp_path, "w", encoding=encoding) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, target)
