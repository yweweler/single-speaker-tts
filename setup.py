# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages


def get_version():
    path = os.path.join(os.path.dirname(__file__), 'tacotron/__init__.py')
    with open(path) as file:
        for line in file:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])


# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
# Import long description
long_description = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='Single-Speaker End-to-End Neural Text-to-Speech Synthesis',
    version=get_version(),
    packages=find_packages(exclude=('tests', 'docs')),

    # PyPI metadata
    author='Yves-Noel Weweler',
    author_email='y.weweler@fh-muenster.de',
    description='This module implements a single-speaker neural text-to-speech (TTS) system capable of training in a end-to-end fashion.',
    long_description=long_description,
    license='MIT License',
    keywords='tensorflow audio neural tts end-to-end single-speaker',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
