# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages


def get_version():
    path = os.path.join(os.path.dirname(__file__), 'conversion/__init__.py')
    with open(path) as file:
        for line in file:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])


# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
# Import long description
long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='voice-conversion',
    version=get_version(),
    packages=find_packages(exclude=('tests', 'docs')),

    # PyPI metadata
    author='Yves-Noel Weweler',
    author_email='y.weweler@fh-muenster.de',
    description='This module implements deep learning based voice conversion.',
    long_description=long_description,
    license='MIT License',
    keywords='mxnet audio autoencoder',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
