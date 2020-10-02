import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transact_dr",
    version="1.0.1",
    author="Soufiane Mourragui <soufiane.mourragui@gmail.com>, ",
    author_email="soufiane.mourragui@gmail.com",
    description="TRANSACT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NKI-CCB/TRANSACT",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn'],
    python_requires='>=3.6',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
    ),
)