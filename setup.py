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
    ],
    entry_points={
        "console_scripts": [
            "prot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["protein_interaction_model.joblib"],
    },
)
