import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="akrule",
    version="v0.0.4",
    author="Hasan Basri Akcay",
    author_email="hasan.basri.akcay@gmail.com",
    description="Highly efficient and precise machine learning models.",
    long_description=(
        "Akrule facilitates rapid and precise predictions, "
        "near real-time training and testing, "
        "comprehensive feature comprehension, "
        "feature debugging, and leakage detection"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/Hasan-Basri-Akcay/akrule",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy", "scipy"],
    keywords=["python", "machine learning", "data science", "exploratory data analysis", "real-time", "beginner"],
)
