from __future__ import annotations

import argparse
import re
from pathlib import Path


FUNCTION_START = "function buildSubjectLikenessConfig("
FUNCTION_END = "// ---------------------------------------------------------------------------\n// Style LoRA"
OLD_ENABLED = "diff_output_preservation: false,"
NEW_ENABLED = "diff_output_preservation: opts.subjectMask,"
OLD_CLASS = "diff_output_preservation_class: 'person',"
NEW_CLASS = "diff_output_preservation_class: opts.subjectMask ? 'woman' : 'person',"
NEW_DESCRIPTION = "'what they caption. Differential Output Preservation is enabled with class woman.',"


def patch_woman_preset(source: str) -> str:
    if source.count(FUNCTION_START) != 1 or source.count(FUNCTION_END) != 1:
        raise RuntimeError("unsupported UI quickstarts: subject-likeness builder anchors changed")
    start = source.index(FUNCTION_START)
    end = source.index(FUNCTION_END, start)
    subject = source[start:end]
    if NEW_ENABLED in subject and NEW_CLASS in subject:
        if NEW_DESCRIPTION not in subject:
            raise RuntimeError("UI quickstart contains an incomplete woman-preset overlay")
        return source
    if subject.count(OLD_ENABLED) != 1 or subject.count(OLD_CLASS) != 1:
        raise RuntimeError("unsupported UI quickstarts: DOP field anchors changed")
    if subject.count("id: 'subject_likeness_masked_flux2_klein9b'") != 1:
        raise RuntimeError("unsupported UI quickstarts: masked preset identity changed")
    patched = subject.replace(OLD_ENABLED, NEW_ENABLED, 1)
    patched = patched.replace(OLD_CLASS, NEW_CLASS, 1)
    description = re.compile(r"(?P<prefix>description:\s*)(?P<quote>['\"])(?P<body>.*?)(?P=quote),", re.DOTALL)
    match = description.search(patched)
    if match is None:
        raise RuntimeError("unsupported UI quickstarts: masked preset description field missing")
    patched = patched[:match.start()] + match.group("prefix") + NEW_DESCRIPTION + patched[match.end():]
    return source[:start] + patched + source[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    quickstarts = args.root / "ui" / "src" / "app" / "jobs" / "new" / "quickstarts.ts"
    if not quickstarts.is_file():
        raise RuntimeError(f"pinned image quickstarts source is missing: {quickstarts}")
    original = quickstarts.read_text(encoding="utf-8")
    patched = patch_woman_preset(original)
    if patched != original:
        quickstarts.write_text(patched, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
