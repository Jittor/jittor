"""Small compiler flag helpers shared by runtime and standalone tools."""


def shsplit(value):
    parts = value.split(" ")
    output = []
    quote_count = 0
    for part in parts:
        current_quotes = part.count('"') + part.count("'")
        if quote_count & 1:
            quote_count += current_quotes
            output[-1] += " " + part
        else:
            quote_count = current_quotes
            output.append(part)
    return output


def remove_flags(flags, removed_prefixes):
    output = []
    for flag in shsplit(flags):
        unquoted = flag.replace('"', "")
        if any(
            unquoted.startswith(prefix) or unquoted.endswith(prefix)
            for prefix in removed_prefixes
        ):
            continue
        output.append(flag)
    return " ".join(output)


__all__ = ["remove_flags", "shsplit"]
