from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="SLIFOX",
    version="1.0.0",
    author="Eldfin (Julian David Wieck)",
    author_email="julian_wieck@yahoo.com",
    license='MIT',
    description="Scattered Light Imaging Fitting (and) Orientation ToolboX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eldfin/SLIF",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.3',  # Python version requirement
    install_requires=[  # List of dependencies
        "numpy>=1.19.5",
        "fcmaes>=1.6.5",
        "h5py>=3.11.0",
        "imageio>=2.34.1",
        "lmfit>=1.3.1",
        "matplotlib>=3.8.4",
        "numba>=0.59.1",
        "numpy>=1.26.4",
        "scipy>=1.13.0",
        "pymp-pypi>=0.5.0",
        "PyQt5>=5.15.10",
        "tqdm>=4.66.4",
        "tifffile>=2024.7.2",
        "nibabel>=5.2.1"
    ]
)
