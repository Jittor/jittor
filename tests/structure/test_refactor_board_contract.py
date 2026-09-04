"""Keep the refactor board aligned with the task plan."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_ROW = re.compile(r"^\|\s*(\d+\.[0-9A-Za-z]+)\s*\|")


def _task_ids(path):
    rows = []
    for line in path.read_text().splitlines():
        match = TASK_ROW.match(line)
        if match:
            rows.append(match.group(1))
    return rows


def test_board_task_ids_match_plan_without_duplicates():
    plan = _task_ids(ROOT / "agent/design/refactor-plan.md")
    board_path = ROOT / "agent/design/refactor-board.md"
    board = _task_ids(board_path)

    assert board == list(dict.fromkeys(board)), "duplicate task rows in board"
    assert set(board) == set(plan)

    for line in board_path.read_text().splitlines():
        if TASK_ROW.match(line):
            assert line.count("|") >= 6, line


def test_board_has_no_two_column_acl_note_rows():
    board = (ROOT / "agent/design/refactor-board.md").read_text().splitlines()
    assert not any(line.startswith("| 8.06 note |") for line in board)
