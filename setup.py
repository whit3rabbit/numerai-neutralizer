from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="numerai-neutralizer",
    version="0.2.0",
    packages=find_packages(exclude=["tests*", "tests.*"]),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0"
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "flake8>=3.8.0"],
    },
    include_package_data=True,
    author="whit3rabbit",
    author_email="whiterabbit@protonmail.com",
    description="Feature neutralization library for Numerai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whit3rabbit/numerai-neutralizer",
    project_urls={
        "Documentation": "https://github.com/whit3rabbit/numerai-neutralizer#readme",
        "Source": "https://github.com/whit3rabbit/numerai-neutralizer",
        "Tracker": "https://github.com/whit3rabbit/numerai-neutralizer/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="numerai, neutralization, machine learning, feature engineering, data science",
    license="MIT",
    python_requires=">=3.8",
)
