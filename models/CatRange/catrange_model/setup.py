"""Setup script for CatRange package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="catrange",
    version="1.2.0",
    author="Abraham Osinuga",
    author_email="oauife.abraham@gmail.com",
    description="Mutation-aware kinetic range prediction for enzymes using CatRange",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TKAI-LAB-Mali/CatRange",
    project_urls={
        "Bug Tracker": "https://github.com/TKAI-LAB-Mali/CatRange/issues",
        "Documentation": "https://github.com/TKAI-LAB-Mali/CatRange",
        "Source Code": "https://github.com/TKAI-LAB-Mali/CatRange",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0,<2",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
        "pyyaml-include>=1.3",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "catrange-train=scripts.cv_train:main",
            "catrange-predict=scripts.predict:main",
            "realkcat-train=scripts.cv_train:main",
            "realkcat-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
