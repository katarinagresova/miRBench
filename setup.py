from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'miRBench',
    version = '0.1.0',
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
        "biopython>=1.79",
        #"requests>=2.23.0",
        #"pip>=20.0.1",
        #"numpy>=1.17.0",
        #"pandas>=1.1.4",
        "viennarna>=2.4.14",
        #'tensorflow>=2.6.0, <2.14.0',
        'torch>=1.9.0',
        'tensorflow==2.13.1',
        'numpy==1.24.3',
        'pandas==2.0.3',
    ],
)
