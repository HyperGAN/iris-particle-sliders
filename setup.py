"""Prevent obsolete profile files in a reused build directory entering a wheel."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class ProfileBuild(build_py):
    def run(self):
        super().run()
        root = Path(self.build_lib)
        for name in ('hndl.py', 'hndl-profile.json', 'installed-profile.json'):
            (root / 'iris_sliders' / name).unlink(missing_ok=True)
        if self.distribution.get_name() == 'iris-particle-sliders':
            shutil.rmtree(root / 'iris_sliders_hndl', ignore_errors=True)


setup(cmdclass={'build_py': ProfileBuild})
