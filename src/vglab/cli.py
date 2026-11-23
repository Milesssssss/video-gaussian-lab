"""已验证的命令：校验清单、打印阻塞、拒绝正式运行、合成深度约束演示。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vglab.demo import synthetic_depth_constraint_demo
from vglab.ingest.manifest import execution_blockers, load_manifest, validate_schema
from vglab.initialization.adapters import environment_blockers
from vglab.paths import example_manifest_path, protocol_path
from vglab.pipeline import run_official
from vglab.protocol import load_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vglab")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("path", nargs="?", default=str(example_manifest_path()))
    validate.add_argument("--require-captured", action="store_true")
    sub.add_parser("show-blockers")
    sub.add_parser("run-official")
    demo = sub.add_parser("demo-depth-constraint")
    demo.add_argument("--steps", type=int, default=24)
    demo.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.command == "validate-manifest":
        return _validate(args.path, args.require_captured)
    if args.command == "show-blockers":
        protocol = load_protocol(protocol_path())
        for item in environment_blockers(protocol):
            print(item)
        return 0
    if args.command == "run-official":
        protocol = load_protocol(protocol_path())
        try:
            run_official(protocol)
        except Exception as exc:
            print(str(exc))
            return 2
        return 2
    if args.command == "demo-depth-constraint":
        return _demo(args.steps, args.output)
    return 2


def _demo(steps: int, output: str | None) -> int:
    if steps <= 0:
        print("--steps 必须为正")
        return 2
    payload = synthetic_depth_constraint_demo(steps=steps)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _validate(path: str, require_captured: bool) -> int:
    manifest = load_manifest(path)
    schema = validate_schema(manifest)
    blockers = execution_blockers(manifest, require_files=require_captured)
    report = {
        "kind": manifest.get("kind"),
        "schema_ok": schema.ok,
        "executable_capture": blockers == [],
        "blockers": blockers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not schema.ok:
        return 2
    if require_captured and blockers:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
