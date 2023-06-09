
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()


def get_version(rel_path):
    for line in (here / rel_path).read_text(encoding="utf-8").splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")



# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")


setup(

    name="torch_multiwell_segmentation", 
    version=get_version("torch_multiwell_segmentation/__init__.py"),
    description="A package for segmenting multiwell plates using PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Jonas Weinz",

    keywords="torch, torchvision, pytorch, deep learning, machine learning, segmentation, multiwell, image processing, image analysis, image segmentation, image classification",

    packages=["torch_multiwell_segmentation"],
    include_package_data=True,
    package_data={"torch_multiwell_segmentation": ["data/model.pt"]},
    python_requires=">=3.7, <4",

    install_requires=[
        "pillow",
        "parse",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "panel",
        "holoviews",
        "bokeh",
        "tqdm",
        "tensorboard"
    ],

    entry_points={
        "console_scripts": [
            "multiwell_dashboard=torch_multiwell_segmentation.serve:main",
        ],
    },

)

