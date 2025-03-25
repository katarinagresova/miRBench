from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'miRBench',
    version = '1.0.1',
    description="A collection of datasets and predictors for benchmarking miRNA target site prediction algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katarina Gresova",
    author_email="gresova11@gmail.com",
    license="MIT",
    keywords=["miRNA", "target site prediction", "benchmarking"],
    url="https://github.com/katarinagresova/miRBench",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.17.0,<2.0",
        "pandas>=1.1.4",
    ],
)
