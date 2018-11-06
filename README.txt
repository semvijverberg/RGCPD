################################################################################
RGCPD
################################################################################

Package to find causal precursors of 1d time series predictant in climate data set. Causal precursors are found based on point-wise correlation maps. The correlation maps generally show regions (spatially co-located gridcells) that significantly correlate. The regions are grouped together to create precursor regions. These precursors regions are subsequently used as masks to create 1d time series. These 1d time series contain more information since they are spatially aggregated. The 1d time series of different precursor regions and subsequently tested for causality using the Tigramite package.  


Installation
************

Installation Anaconda3, python 3.6.1 environment

# Installing in conda base_root 
conda install spyder=3.3.1
conda config --add channels conda-forge
conda config --append channels bioconda
conda install numpy pandas matplotlib cartopy xarray netCDF4 ecmwfapi scipy seaborn netcdftime cyordereddict
conda install -c conda-forge nco


- If you want to install this package in an environment this will not work since it will automatically install Tigramite to your root site-package e.g. ~anaconda/lib/python2.7/site-packages/. And running Spyder from source activate <env> will point to the site-packages in your env folder ~/anaconda/envs/<env_name>/lib/python3.6/site-packages, not your root (thus it will not find the tigramite module). 
To solve this you could do:

# Installing in conda env 
Easy way:
conda create --name RGCPDspec --file conda_env.txt 
More typing way, but probably cleaner:
conda config --add channels conda-forge
conda config --append channels bioconda
conda create --name <env_name> python=3.6 numpy pandas matplotlib cartopy xarray netCDF4 ecmwfapi scipy seaborn netcdftime cyordereddict pip



# ECWMF MARS API
If you did not have ecmwfapi installed before, you need to create an ecmwf account and copy your key into the file .ecmwfapirc in your home directory. This will look like this:
{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : <your key>,
    "email" : <your emailadress>
}


# Installing Tigramite
To install Tigramite, see https://github.com/jakobrunge/tigramite
- If you install Tigramite on your base_root in conda, it will satisfy to download the tigramite-master folder and run 
python setup.py install

# Installing Tigramite in conda env:
Source activate <env_name>
Git clone https://github.com/jakobrunge/tigramite.git 
pip install ./tigramite 

# Installing CDO 

cdo -V
Climate Data Operators version 1.9.4 (http://mpimet.mpg.de/cdo)
System: x86_64-apple-darwin17.6.0
CXX Compiler: /usr/bin/clang++ -std=gnu++11 -pipe -Os -stdlib=libc++ -isysroot/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -arch x86_64  -D_THREAD_SAFE -pthread
CXX version : unknown
C Compiler: /usr/bin/clang -pipe -Os -isysroot/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -arch x86_64  -D_THREAD_SAFE -pthread -D_THREAD_SAFE -D_THREAD_SAFE -pthread
C version : unknown
F77 Compiler:  -pipe -Os
Features: 16GB C++11 DATA PTHREADS HDF5 NC4/HDF5 OPeNDAP UDUNITS2 PROJ.4 CURL FFTW3 SSE4_1
Libraries: HDF5/1.10.2 proj/5.1 curl/7.60.0
Filetypes: srv ext ieg grb1 grb2 nc1 nc2 nc4 nc4c nc5 
     CDI library version : 1.9.4
GRIB_API library version : 2.7.0
  NetCDF library version : 4.4.1.1 of Jun  8 2018 03:07:16 $
    HDF5 library version : 1.10.2
    EXSE library version : 1.4.0
    FILE library version : 1.8.3


Project Setup
*************

Here we provide some details about the project setup. Most of the choices are explained in the `guide <https://guide.esciencecenter.nl>`_. Links to the relevant sections are included below.
Feel free to remove this text when the development of the software package takes off.

For a quick reference on software development, we refer to `the software guide checklist <https://guide.esciencecenter.nl/best_practices/checklist.html>`_.

Version control
---------------

Once your Python package is created, put it under
`version control <https://guide.esciencecenter.nl/best_practices/version_control.html>`_!
We recommend using `git <http://git-scm.com/>`_ and `github <https://github.com/>`_.

.. code-block:: console

  cd RGCPD
  git init
  git add -A
  git commit

To put your code on github, follow `this tutorial <https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line/>`_.

Python versions
---------------

This repository is set up with Python versions:
* 3.4
* 3.5
* 3.6

Add or remove Python versions based on project requirements. `The guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html>`_ contains more information about Python versions and writing Python 2 and 3 compatible code.

Package management and dependencies
-----------------------------------

You can use either `pip` or `conda` for installing dependencies and package management. This repository does not force you to use one or the other, as project requirements differ. For advice on what to use, please check `the relevant section of the guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#dependencies-and-package-management>`_.

* Dependencies should be added to `setup.py` in the `install_requires` list.

Packaging/One command install
-----------------------------

You can distribute your code using pipy or conda. Again, the project template does not enforce the use of either one. `The guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#building-and-packaging-code>`_ can help you decide which tool to use for packaging.

Testing and code coverage
-------------------------

* Tests should be put in the ``tests`` folder.
* The testing framework used is `PyTest <https://pytest.org>`_

  - `PyTest introduction <http://pythontesting.net/framework/pytest/pytest-introduction/>`_

* Tests can be run with ``python setup.py test``

  - This is configured in ``setup.py`` and ``setup.cfg``

* Use `Travis CI <https://travis-ci.com/>`_ to automatically run tests and to test using multiple Python versions

  - Configuration can be found in ``.travis.yml``
  - `Getting started with Travis CI <https://docs.travis-ci.com/user/getting-started/>`_

* TODO: add something about code quality/coverage tool?
* `Relevant section in the guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#testing>`_

Documentation
-------------

* Documentation should be put in the ``docs`` folder. The contents have been generated using ``sphinx-quickstart`` (Sphinx version 1.6.5).
* We recommend writing the documentation using Restructured Text (reST) and Google style docstrings.

  - `Restructured Text (reST) and Sphinx CheatSheet <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_
  - `Google style docstring examples <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

* To generate html documentation run ``python setup.py build_sphinx``

  - This is configured in ``setup.cfg``
  - Alternatively, run ``make html`` in the ``docs`` folder.

* The ``docs/_static`` and ``docs/_templates`` contain an (empty) ``.gitignore`` file, to be able to add them to the repository. These two files can be safely removed (or you can just leave them there).
* To put the documentation on `Read the Docs <https://readthedocs.org>`_, log in to your Read the Docs account, and import the repository (under 'My Projects').

  - Include the link to the documentation in this README_.

* `Relevant section in the guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#writingdocumentation>`_

Coding style conventions and code quality
-----------------------------------------

* Check your code style with ``prospector``
* You may need run ``pip install .[dev]`` first, to install the required dependencies
* You can use ``yapf`` to fix the readability of your code style and ``isort`` to format and group your imports
* `Relevant section in the guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#coding-style-conventions>`_

CHANGELOG.rst
-------------

* Document changes to your software package
* `Relevant section in the guide <https://guide.esciencecenter.nl/software/releases.html#changelogmd>`_

CITATION.cff
------------

* To allow others to cite your software, add a ``CITATION.cff`` file
* It only makes sense to do this once there is something to cite (e.g., a software release with a DOI).
* To generate a CITATION.cff file given a DOI, use `doi2cff <https://github.com/citation-file-format/doi2cff>`_.
* `Relevant section in the guide <https://guide.esciencecenter.nl/software/documentation.html#citation-file>`_

CODE_OF_CONDUCT.rst
-------------------

* Information about how to behave professionally
* `Relevant section in the guide <https://guide.esciencecenter.nl/software/documentation.html#code-of-conduct>`_

CONTRIBUTING.rst
----------------

* Information about how to contribute to this software package
* `Relevant section in the guide <https://guide.esciencecenter.nl/software/documentation.html#contribution-guidelines>`_

MANIFEST.in
-----------

* List non-Python files that should be included in a source distribution
* `Relevant section in the guide <https://guide.esciencecenter.nl/best_practices/language_guides/python.html#building-and-packaging-code>`_

NOTICE
------

* List of licenses of the project and dependencies
* `Relevant section in the guide <https://guide.esciencecenter.nl/best_practices/licensing.html#notice>`_

Installation
------------

To install RGCPD, do:

.. code-block:: console

  git clone https://github.com//RGCPD.git
  cd RGCPD
  pip install .


Run tests (including coverage) with:

.. code-block:: console

  python setup.py test


Documentation
*************

.. _README:

Include a link to your project's full documentation here.

Contributing
************

If you want to contribute to the development of RGCPD,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2018, VU Amsterdam

GNU General Public License v3

Credits
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
