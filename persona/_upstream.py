"""Bridge to upstream `src/steering.py` so we can reuse it unchanged.

Upstream's scripts assume the repo root is on the path and `import steering`
just works. As an installable package we can't rely on that, so we inject
`<repo>/src` onto sys.path here and re-export the bits we need.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from steering import generate_steered_answer  # noqa: E402, F401
except ImportError as e:
    raise ImportError(
        "Could not import upstream src/steering.py. "
        "Make sure you're running from the eiffel-steering repo root, or that "
        f"{_SRC} exists."
    ) from e
