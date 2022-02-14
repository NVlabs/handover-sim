import sys
import subprocess

from setuptools.command.develop import develop as _develop
from setuptools import setup


class develop(_develop):

  def run(self):
    _develop.run(self)

    # Ideally, one should be able to also install the dependent Python packages
    # resided in submodules with one pip install run from the main repo. This is
    # possible using `install_requires` with file URLs:
    # * https://stackoverflow.com/questions/28113862/how-to-install-a-dependency-from-a-submodule-in-python
    # * https://stackoverflow.com/questions/64878322/setuptools-find-package-and-automatic-submodule-dependancies-management
    # * https://stackoverflow.com/questions/64988110/using-git-submodules-with-python
    # * https://www.python.org/dev/peps/pep-0440/#direct-references
    # * https://github.com/pypa/pip/issues/6658#issuecomment-506841157
    #
    # For example:
    #
    #   install_requires=[
    #       f'mano_pybullet @ file://{os.getcwd()}/mano_pybullet',
    #   ]
    #
    # However, it is currently not possible to use this method for editable
    # (develop) mode.
    # * https://stackoverflow.com/questions/68491950/specifying-develop-mode-for-setuptools-setup-install-requires-argument
    #
    # Use the subprocess hack. This also allows running with `--no-deps`.
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--verbose', '--no-cache-dir',
        '--no-deps', '--editable', 'mano_pybullet'
    ])


setup(
    name='handover-sim',
    cmdclass={'develop': develop},
    install_requires=[
        'chumpy==0.70',
        'gym==0.21.0',
        'numpy==1.22.2',
        'pybullet==3.2.1',
        'PyYAML==6.0',
        'scipy==1.8.0',
        'transforms3d==0.3.1',
        'yacs==0.1.8',
    ],
)
