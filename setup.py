from setuptools import setup, find_packages

setup(
    name='namd',
    version='0.1',
    packages=find_packages(where='src'),  # Look for packages inside the 'src' directory
    package_dir={'': 'src'},              # Root directory for packages is 'src'
)

