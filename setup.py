import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finance_data_pull",
    version="0.0.1",
    author="Cedd Burge",
    author_email="ceddlyburge@gmail.com",
    description="A function that returns 'world'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mat2698/returns_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)