"""Setuptools compatibility entry point for the OpenDPD-TCN-QAT fork."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="opendpd-tcn-qat",
    version="2.1.0",
    author="OpenDPD Authors and DPD-Flow Contributors",
    author_email="chang.gao@tudelft.nl",
    description=(
        "OpenDPD fork with native full-I/O causal-TCN QAT and portable RTL "
        "export"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/chldjwls85/OpenDPD-TCN-QAT",
    project_urls={
        "Documentation": (
            "https://github.com/chldjwls85/OpenDPD-TCN-QAT/blob/"
            "tcn-qat/docs/TCN_QAT_WORKFLOW.md"
        ),
        "Upstream": "https://github.com/lab-emi/OpenDPD",
    },
    packages=find_packages(exclude=["Matlab", "pics", "slprj"]),
    py_modules=["arguments", "project", "models", "main"],
    include_package_data=True,
    package_data={
        "datasets": ["*/spec.json", "*/*.csv", "*/*.py"],
        "quant": ["schemas/*.json"],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "numpy>=2.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pillow>=9.0.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "build>=1.0",
            "twine>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "opendpd-cli=opendpd.cli:main",
            "opendpd-rtl-export=quant.rtl_cli:export_main",
            "opendpd-rtl-verify=quant.rtl_cli:verify_main",
            "opendpd-rtl-evaluate=scripts.evaluate_fexlite_integer_pa:main",
        ],
    },
)
