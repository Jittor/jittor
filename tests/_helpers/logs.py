"""Log-capture helpers shared by compiler and backend tests."""

import re


def find_log_with_re(logs, pattern=None, **args):
    if pattern:
        pattern = re.compile(pattern)
    filtered = []
    for log in logs:
        for name in args:
            if log[name] != args[name]:
                break
        else:
            if pattern:
                matches = re.findall(pattern, log["msg"])
                if matches:
                    filtered.append(matches[0])
            else:
                filtered.append(log["msg"])
    return filtered
