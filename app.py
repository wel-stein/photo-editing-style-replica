"""Photo Style AI entry point.

Default: launches the Gradio UI on http://127.0.0.1:7860.
Pass `--no-ui` followed by CLI args to bypass the UI and use the same
training/apply/export commands as `python -m photo_style.cli`.

Examples:
    python app.py
    python app.py --no-ui train --originals NEFs --edited edits --name my_style
    python app.py --no-ui apply --profile my_style --input shot.NEF --output out.jpg
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if "--no-ui" in argv:
        argv = [a for a in argv if a != "--no-ui"]
        from photo_style.cli import main as cli_main

        return cli_main(argv)
    from photo_style.ui import launch

    return launch()


if __name__ == "__main__":
    sys.exit(main())
