from setuptools import setup, find_packages

setup(
    name = 'miRBench',
    packages=find_packages("src"),
    package_dir={"": "src"},
)