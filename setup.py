import sys
import subprocess

from setuptools.command.develop import develop as _develop
from setuptools import setup


# The current implementation requires a dependency of pybullet with version
# >=3.0.9 and <=3.1.2, due to a memory issue on `resetSimulation()` with EGL:
# * https://github.com/bulletphysics/bullet3/issues/3285
# * https://github.com/bulletphysics/bullet3/commit/910c20334f9f435b292fd1d51ec2b57bfa90eda9
#
# This is a hack to pip install pybullet==3.1.2 with numpy enabled, since
# listing it under install_requires will not trigger a numpy build if numpy has
# not been installed already:
# * https://stackoverflow.com/questions/40831794/call-another-setup-py-in-setup-py
# * https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
# * https://niteo.co/blog/setuptools-run-custom-code-in-setup-py
class develop(_develop):

  def run(self):
    # Run pip install in a Python process:
    # * https://stackoverflow.com/questions/12332975/installing-python-module-within-code
    # * https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--verbose', '--no-cache-dir',
        'pybullet==3.1.2'
    ])

    _develop.run(self)


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
    ],
)
