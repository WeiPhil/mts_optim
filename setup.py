from setuptools import setup

# Setup mts_optim library
setup(
    name="mts_optim",
    version="1.0",
    author="Philippe Weier",
    description="",
    packages=["mts_optim"],
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'matplotlib',
        'gin-config==0.5.0',
        'tqdm',
        'mediapy==1.2.2',
        'colorcet==3.1.0',
        'scipy==1.15.2',
        'cholespy==2.1.0',
        'ipywidgets',
        'ipykernel',
    ],
)
