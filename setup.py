#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='RGCPD',
    version='0.1',
    description="Package to find causal precursors of predictant in climate data set",
    long_description=readme + '\n\n',
    author="Sem Vijverberg ",
    author_email='sem.vijverberg@vu.nl',
    url='https://github.com//RGCPD',
    packages=[
        'RGCPD',
    ],
    package_dir={'RGCPD':
                 'RGCPD'},
    include_package_data=True,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='RGCPD',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    install_requires=[],  # FIXME: add your package's dependencies to this list
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
