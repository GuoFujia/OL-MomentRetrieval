from setuptools import find_packages, setup

setup(
    name="ol-moment-retrieval",
    version="0.1.0",
    description="Official inference implementation for OL-MomentRetrieval models",
    python_requires=">=3.9",
    install_requires=[
        "easydict",
        "einops",
        "ffmpeg-python",
        "ftfy",
        "fvcore",
        "librosa",
        "msclap",
        "pandas",
        "PyYAML",
        "regex",
        "scikit-learn",
        "torch",
        "torchlibrosa",
        "tqdm",
    ],
    packages=find_packages(),
)
