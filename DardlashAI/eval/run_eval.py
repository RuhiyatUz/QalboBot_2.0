# -*- coding: utf-8 -*-
"""Печатает 30 кейсов для ручной оценки по eval/rubric.md."""
import json
from pathlib import Path

CASES = Path(__file__).with_name("cases.json")


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    print(f"Кейсов: {len(cases)}. Рубрика: eval/rubric.md\n")
    for case in cases:
        print(f"[{case['id']:02d}] lang={case['lang']} mode={case['mode']}")
        print(f"     {case['text']}\n")


if __name__ == "__main__":
    main()
