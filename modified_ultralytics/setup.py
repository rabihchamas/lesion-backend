from setuptools import setup, find_packages

setup(
    name="ultralytics_lesions",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # ← this enables MANIFEST.in usage
    package_data={
        "ultralytics_lesions": ["cfg/*.yaml"],  # adjust path as needed
    },
)
