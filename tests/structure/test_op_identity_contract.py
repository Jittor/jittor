import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "python" / "jittor" / "src"


def test_operator_identity_does_not_use_base_name_strings():
    forbidden = re.compile(
        r"(?<![A-Za-z0-9_])name\(\)\s*(?:==|!=)|"
        r"(?:string\([^)]*\)|\"[^\"]*\")\s*(?:==|!=)[^;\n]*"
        r"(?<![A-Za-z0-9_])name\(\)|"
        r"strcmp\([^,\n]*name\(\)|fast_strcmp"
    )
    offenders = []
    for path in sorted(SRC.rglob("*")):
        if path.suffix not in (".cc", ".h"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                offenders.append("{}:{}:{}".format(path.relative_to(REPO_ROOT), lineno, line.strip()))
    assert not offenders, "operator identity still uses strings:\n" + "\n".join(offenders)
