from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'miRBench',
    version = '1.0.3',
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
        "numpy>=1.17.0",
        "pandas>=1.1.4",
    ],
    extras_require={
        # Dependencies of encoders and predictors. The miRBenchCNN models were saved by Keras 2.13,
        # whose .keras files later Keras versions cannot load (TensorFlow 2.13 needs Python 3.8 - 3.11).
        "models": ["biopython", "viennarna", "torch", "tensorflow~=2.13.0"],
        # The exact versions the predictors were validated with (all have wheels for Python 3.9 only).
        "exactmodels": [
            "numpy==1.24.3",
            "biopython==1.83",
            "viennarna==2.7.0",
            "torch==1.9.0",
            "tensorflow==2.13.1",
            "typing-extensions==4.5.0",
        ],
    },
)
