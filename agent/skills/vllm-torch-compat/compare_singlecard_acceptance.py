"""Compare completed single-card Qwen runs; report stochastic differences explicitly."""
import argparse
import json
from pathlib import Path


def compare(root):
    def read(backend, case):
        data = json.loads((root / (backend + "-" + case + ".json")).read_text())
        assert data["status"] == "completed", (backend, case)
        return data

    def outputs(data):
        return [(row["label"], output) for row in data["results"] for output in row["outputs"]]

    summary = {}
    for case in ("state", "random", "long"):
        a, b = (read(backend, case) for backend in ("jittor", "oracle"))
        assert a["options"] == b["options"], case
        left, right = outputs(a), outputs(b)
        assert len(left) == len(right), case
        rows = []
        for (label, x), (other_label, y) in zip(left, right):
            assert label == other_label, (label, other_label)
            assert x.get("prompt_token_ids", x.get("prompt_tokens")) == y.get("prompt_token_ids", y.get("prompt_tokens")), label
            assert len(x["token_ids"]) == len(y["token_ids"]), label
            greedy = case == "long" or (case == "state" and x["temperature"] == 0)
            exact = x["token_ids"] == y["token_ids"]
            if greedy:
                assert exact, (case, label, "greedy mismatch")
            first_difference = next((i for i, (u, v) in enumerate(zip(x["token_ids"], y["token_ids"])) if u != v), None)
            rows.append(dict(label=label, greedy=greedy, exact=exact,
                             output_tokens=len(x["token_ids"]), first_difference=first_difference))
        if case == "state":
            assert a["assertions"] and b["assertions"]
            assert all(a["assertions"].values()) and all(b["assertions"].values())
        if case == "random":
            assert a["seed_reproducible"] and b["seed_reproducible"]
            assert a["seed_diversity"] > 1 and b["seed_diversity"] > 1
        summary[case] = dict(requests=len(rows), output_tokens_per_backend=sum(r["output_tokens"] for r in rows),
                             exact_requests=sum(r["exact"] for r in rows),
                             greedy_requests=sum(r["greedy"] for r in rows),
                             greedy_tokens=sum(r["output_tokens"] for r in rows if r["greedy"]),
                             rows=rows, jittor_default=a["default_device"], oracle_default=b["default_device"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    (args.directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for case, data in result.items():
        print(case, {key: value for key, value in data.items() if key != "rows"})
