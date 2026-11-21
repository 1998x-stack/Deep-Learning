"""Compatibility entry point: validate the published Deep-Learning site.

The former script regenerated README.md and a hard-coded directory tree.
That destructive behavior is intentionally removed.
"""
from scripts.validate_site import main
if __name__ == "__main__":
    raise SystemExit(main())
