"""Run the installed native wheel on real Windows with the selected MSVC toolset."""

import argparse
from email.parser import BytesParser
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from zipfile import ZipFile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.name != 'nt' or platform.system() != 'Windows':
        parser.error('Windows/MSVC verification requires a real Windows process')
    if not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('--commit must be the full measured source SHA')
    compiler = shutil.which('cl.exe')
    if not compiler:
        parser.error('cl.exe is missing; activate the MSVC x64 developer environment')
    with ZipFile(args.wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise RuntimeError('expected one wheel metadata entry')
        expected = BytesParser().parsebytes(archive.read(metadata[0]))
    installed = distribution('jittor')
    if expected['Name'] != 'jittor' or installed.version != expected['Version']:
        raise RuntimeError('installed distribution does not match the selected Jittor wheel')
    record = dict(status='running', source_commit=args.commit, os=platform.platform(),
                  python=sys.version, compiler=compiler, toolset=os.environ.get('VCToolsVersion'),
                  wheel=args.wheel.name, wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                  installed_version=installed.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

    # Import only after checking the host: Linux syntax tests cannot certify
    # the Windows DLL loader, compiler, or installed-package layout.
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks
    from jittor.selftest import run

    package = Path(jt.__file__).resolve()
    checkout = args.checkout.resolve()
    installed_package = Path(installed.locate_file('jittor')).resolve()
    if checkout == package or checkout in package.parents:
        raise RuntimeError('verification imported the source checkout instead of the wheel')
    if installed_package not in package.parents:
        raise RuntimeError('runtime package is outside the installed wheel distribution')
    if jt.compiler.cc_type != 'cl' or jt.flags.use_cuda or jt.compiler.has_acl:
        raise RuntimeError('verification requires the actual MSVC CPU build')
    with forbid_backend_fallbacks():
        result = run()
        jt.sync_all(True)
    if result['backend'] != 'cpu':
        raise RuntimeError('installed self-test ran on an unexpected backend')
    record.update(status='passed', package=str(package), core=str(jt.core.__file__),
                  compiler_type=jt.compiler.cc_type, selftest=result,
                  fallback_count=jt.core.backend_fallback_count())
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
