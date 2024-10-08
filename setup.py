from setuptools import setup, find_packages

setup(
    name="prot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "pandas",
        "matplotlib",
        "seaborn",
        "imbalanced-learn",
        "optuna",
    ],
    entry_points={
        "console_scripts": [
            "prot=main:main",
        ],
    },
    include_package_data=True,
    # Remove the package_data parameter
)
