from setuptools import find_packages, setup


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description


def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="ailab-llama-search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
)
