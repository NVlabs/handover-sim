import sys
import subprocess

from setuptools.command.develop import develop as _develop
from setuptools import setup


class develop(_develop):

  def run(self):
    # The current implementation requires a dependency of pybullet with version
    # >=3.0.9 and <=3.1.2, due to a memory issue on `resetSimulation()` with EGL:
    # * https://github.com/bulletphysics/bullet3/issues/3285
    # * https://github.com/bulletphysics/bullet3/commit/910c20334f9f435b292fd1d51ec2b57bfa90eda9
    #
    # This is a hack to pip install pybullet==3.1.2 with numpy enabled, since
    # listing it under `install_requires` will not trigger a numpy build if numpy
    # has not been installed already:
    # * https://stackoverflow.com/questions/40831794/call-another-setup-py-in-setup-py
    # * https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
    # * https://niteo.co/blog/setuptools-run-custom-code-in-setup-py
    #
    # Run pip install in a Python process:
    # * https://stackoverflow.com/questions/12332975/installing-python-module-within-code
    # * https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--verbose', '--no-cache-dir',
        'pybullet==3.1.2'
    ])

    _develop.run(self)

    # Ideally, one should be able to also install the dependant Python packages
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
        '--no-deps', '--editable', 'mano_pybullet/'
    ])


setup(
    name='handover-sim',
    cmdclass={'develop': develop},
    install_requires=[
        'chumpy==0.70',
        'easydict==1.9',
        'gym==0.18.0',
        'numpy==1.19.5',
        'pyyaml==5.4.1',
        'scipy==1.6.2',
        'transforms3d==0.3.1',
    ],
)
