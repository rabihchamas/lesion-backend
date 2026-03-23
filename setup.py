from setuptools import setup, find_packages

setup(
    name="lesion-backend",
    version="0.1.0",
    packages=find_packages(exclude=["modified_ultralytics*"]),
)
