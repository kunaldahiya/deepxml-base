import setuptools

setuptools.setup(
    name="deepxml",
    version="0.1",
    author="X",
    author_email="kunalsdahiya@gmail.com",
    description="An deep extreme classification library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/kunaldahiya/deepxml-base",
    install_requires=['numpy', 'torch'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
