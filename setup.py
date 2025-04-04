from setuptools import setup, find_packages

setup(
    name="cpu_hybrid_recommender",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "implicit>=0.4.0",
        "joblib>=1.0.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "annoy": ["annoy>=1.17.0"],
        "nmslib": ["nmslib>=2.1.0"],
        "all": [
            "annoy>=1.17.0",
            "nmslib>=2.1.0",
        ],
    },
    python_requires=">=3.7",
    description="CPU-optimized hybrid recommendation system",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/cpu_hybrid_recommender",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "train-recommender=src.train:main",
        ],
    },
)