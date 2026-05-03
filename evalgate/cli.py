import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evalgate import gate
from evalgate.config import GateConfig
from evalgate.errors import EvalgateError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='evalgate')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='run the pyramid and exit 1 on a regression')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--baseline-dir', type=Path, required=True)
    run.add_argument('--candidate-dir', type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = GateConfig.load(args.config)
        code = gate.run(config, args.baseline_dir, args.candidate_dir)
    except EvalgateError as error:
        print(f'evalgate error, {error}', file=sys.stderr)
        return 2
    report_path = args.candidate_dir / gate.REPORT_DIR / 'report.md'
    print(f'{"pass" if code == 0 else "block"}, report at {report_path}')
    return code
