from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['gym>=0.9.1']
setup(
    name='cartpole',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='kmlcc-training'
)
