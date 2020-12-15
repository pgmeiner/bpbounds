import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="bpbounds-PGMEINER",
    version="0.0.1",
    author="Peter Gmeiner",
    author_email="peter.gmeiner@algobalance.com",
    description="Nonparametric bound calculation for the average treatment effect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgmeiner/bpbounds",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE V3 (GPLV3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=required)
