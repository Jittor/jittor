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
    plan = _task_ids(ROOT / "docs/architecture/refactor-plan.md")
    board_path = ROOT / "docs/architecture/refactor-board.md"
    board = _task_ids(board_path)

    assert board == list(dict.fromkeys(board)), "duplicate task rows in board"
    assert set(board) == set(plan)

    for line in board_path.read_text().splitlines():
        if TASK_ROW.match(line):
            cells = re.split(r"(?<!\\)\|", line)
            assert len(cells) == 7, line
            assert not cells[0].strip() and not cells[-1].strip(), line


def test_board_has_no_two_column_acl_note_rows():
    board = (ROOT / "docs/architecture/refactor-board.md").read_text().splitlines()
    assert not any(line.startswith("| 8.06 note |") for line in board)


def test_functional_audit_covers_every_open_ledger_row():
    """The audit must distinguish functional evidence from closure criteria."""
    audit = (ROOT / "docs/results/2026-09-09-functional-board-audit.md").read_text()
    matrix = audit.split("## Functional closure matrix", 1)[1]
    for task_id in ("0.15", "0.22", "2.19", "3.20", "3.22", "3.23",
                    "8.05", "8.06", "8.21"):
        assert f"| {task_id} |" in matrix
    assert "| row | implementation | functional verification | performance | hardware |" in matrix
    assert "Only the performance and hardware columns" in matrix
