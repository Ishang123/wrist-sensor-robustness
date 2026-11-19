"""
Render README.md from README.template.md by inlining generated result tables.

Placeholders look like {{tables/baseline.md}} and are replaced with the file's
contents from results/. Every table in the README therefore comes straight
from a file the analysis code wrote - none are typed by hand. Fails loudly if
a referenced table is missing.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    template = (ROOT / "README.template.md").read_text()

    def inline(match):
        path = ROOT / "results" / match.group(1)
        if not path.exists():
            sys.exit(f"missing {path} - run `make analyze` first")
        return path.read_text().rstrip()

    out = re.sub(r"\{\{([^}]+)\}\}", inline, template)
    (ROOT / "README.md").write_text(out)
    print(f"README.md written ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
