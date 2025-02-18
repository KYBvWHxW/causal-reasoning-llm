from setuptools import setup, find_packages

setup(
    name="causal-reasoning-llm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "psutil>=5.9.0",
            "locust>=2.15.0",
        ],
    },
)
