"""
This is the setup module for the HSTransform package. (Beta test)

S-transform with Hyperbolic Window
Combines the best properties of the Short-time Fourier Transform 
and the Wavelet Transform.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='HSTransform',
    version='0.1.1',
    packages=find_packages(),
    license='MIT',
    author='Linh V Nguyen',
    author_email='linhvietnguyen.ee@gmail.com',
    description='A Package to Compute S-transform with Hyperbolic Window',
    python_requires='>=3.10',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests'
)
