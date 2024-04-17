from setuptools import setup, find_packages

setup(
    name = 'miRNAbenchmarks',
    packages=find_packages("src"),
    package_dir={"": "src"},
)