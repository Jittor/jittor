"""Parser helpers for tuner diagnostic output."""


def simple_parser(value):
    parts = value.split(":")
    result = []
    for item in parts[:-1]:
        end = len(item) - 1
        if end < 0:
            result.append("")
            continue
        while end >= 0 and item[end] in " \n":
            end -= 1
        start = end
        while start >= 0 and item[start] not in " \n{},":
            start -= 1
        result.append(
            '{}"{}"{}'.format(item[: start + 1], item[start + 1 : end + 1], item[end + 1 :])
        )
    result.append(parts[-1])
    return eval(":".join(result))
