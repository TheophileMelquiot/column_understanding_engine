from setuptools import setup, find_packages

setup(
    name="column_engine",
    version="0.1.0",
    description="Column Understanding Engine — Semantic understanding of tabular data from raw Excel files",
    author="Théophile Melquiot",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "openpyxl>=3.1.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
