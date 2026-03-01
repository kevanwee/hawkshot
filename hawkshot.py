"""Compatibility entrypoint for Hawkshot CLI."""

from hawkshot.cli import main


if __name__ == "__main__":
    raise SystemExit(main())