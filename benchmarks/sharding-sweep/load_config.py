"""Minimal YAML-subset reader for run.sh defaults (stdlib only, no PyYAML).

Emits TAB-separated lines that run.sh consumes:
  <key>\t<value>                       for each top-level scalar
  MODEL\t<tag>\t<path>\t<label>        for each item under `models:` (label optional)

This is NOT a general YAML parser. It supports exactly the flat-scalar +
`models:` list schema documented in config.example.yaml, so it can run under a
bare `python3` (no third-party deps) and even bootstrap the venv that is itself
named in the config. Unknown keys are passed through; run.sh ignores ones it
doesn't recognize.

usage: python3 load_config.py config.yaml
"""
import sys


def strip_comment(s):
    """Drop a trailing # comment that is outside single/double quotes."""
    out, quote = [], None
    for ch in s:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: load_config.py config.yaml")
    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()

    out = []
    in_models = False
    cur = None  # current model item dict

    def flush():
        nonlocal cur
        if cur and cur.get("tag") and cur.get("path"):
            out.append("MODEL\t%s\t%s\t%s"
                       % (cur["tag"], cur["path"], cur.get("label", "")))
        cur = None

    for raw in lines:
        line = strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            flush()
            in_models = False
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = unquote(val.strip())
            if key == "models" and val == "":
                in_models = True
                continue
            if key:
                out.append("%s\t%s" % (key, val))
            continue

        # indented line
        if not in_models:
            continue
        if stripped.startswith("-"):
            flush()
            cur = {}
            stripped = stripped[1:].strip()
        if cur is None:
            cur = {}
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            cur[k.strip()] = unquote(v.strip())

    flush()
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
