"""Extract specific triton kernel bodies from the Inductor cache dump for inspection.

Usage:
  python scripts/extract_kernel_body.py <kernel_name> [--cache path]
"""
import argparse, pathlib, re, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="Kernel names (regex-escaped prefixes OK)")
    parser.add_argument("--cache", default="docs/perf/inductor-triton-dump.cache.py")
    parser.add_argument("--out", default=None, help="Output file (default: stdout)")
    parser.add_argument("--max-bytes", type=int, default=8000,
                        help="Max body bytes per kernel to print")
    args = parser.parse_args()

    text = pathlib.Path(args.cache).read_text(encoding="utf-8", errors="replace")

    buf = []
    for name in args.names:
        pattern = re.compile(
            r"async_compile\.triton\(\s*['\"]" + re.escape(name) + r"['\"]\s*,\s*'''(.*?)'''",
            re.S,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            buf.append(f"# NOT FOUND: {name}\n")
            continue
        for i, m in enumerate(matches):
            body = m.group(1)
            buf.append(f"\n\n=================================================================")
            buf.append(f"# KERNEL: {name}  (instance {i+1}/{len(matches)}, {len(body)} bytes)")
            buf.append(f"=================================================================\n")
            buf.append(body[: args.max_bytes])
            if len(body) > args.max_bytes:
                buf.append(f"\n# ... truncated ({len(body) - args.max_bytes} bytes remaining)\n")

    out_text = "\n".join(buf)
    if args.out:
        pathlib.Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"Wrote {args.out} ({len(out_text)} bytes)")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
