import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="speednet", # Replace with your own username
    version="0.0.0",
    author="Abhranta Panigrahi",
    author_email="abhranta.panigrahir@example.com",
    description="This is a package that aims to make various neural networks accessbile to a wider audience by providing black box models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Abhranta/speednet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data = True , 
    install_requires = ["torch >= 1.6.0"]
)