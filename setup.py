"""
Setup script for hypertrack package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="hypertrack",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for tracking Hyperliquid trading activity and detecting profitable traders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hypertrack",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hypertrack/issues",
        "Source": "https://github.com/yourusername/hypertrack",
        "Documentation": "https://github.com/yourusername/hypertrack#readme",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hypertrack=hypertrack.recorder:main",
        ],
    },
    keywords="hyperliquid, trading, cryptocurrency, defi, blockchain, trading-bot, ml, machine-learning",
    include_package_data=True,
    zip_safe=False,
)
