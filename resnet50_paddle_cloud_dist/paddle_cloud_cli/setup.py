"""
This package provides the paddlecloud setup command.
"""
from distutils.core import setup
import distutils.sysconfig
import os
import shutil

package_name = "paddlecli"
install_target_path = os.path.join(distutils.sysconfig.get_python_lib(), package_name)
if os.path.exists(install_target_path):
    shutil.rmtree(install_target_path)

setup(
    name='paddleplatform',
    version='0.10.0',
    packages=['paddlecli',
              'paddlecli.core',
              'paddlecli.core.http',
              'paddlecli.lib',
              'paddlecli.conf',
              'paddlecli.argparser',
              ],
    scripts=['paddlecloud'],
    url='',
    license='',
    author='',
    author_email='',
    description=''
)
