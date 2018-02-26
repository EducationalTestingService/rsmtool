.. _install:

Installation
============
Note that RSMTool has only been tested with Python 3.6 and higher. 

Installing with conda
----------------------

Currently, the recommended way to install RSMTool is by using the ``conda`` package manager. If you have already installed ``conda``, you can skip straight to Step 2.

1. To install ``conda``, follow the instructions on `this page <http://conda.pydata.org/docs/install/quick.html>`_.

2. Create a new conda environment (say, ``rsmtool``) and install the RSMTool conda package by running::

    conda create -n rsmtool -c defaults -c conda-forge -c desilinguist python=3.6 rsmtool

3. Activate this conda environment by running ``source activate rsmtool`` (``activate rsmtool`` on windows). You should now have all of the RSMTool command-line utilities in your path.

4. From now on, you will need to activate this conda environment whenever you want to use RSMTool. This will ensure that the packages required by RSMTool will not affect other projects.

RSMTool can also be downloaded directly from
`GitHub <http://github.com/EducationalTestingService/rsmtool>`_.

Installing with pip
-------------------

You can also ``pip`` to install RSMTool instead of ``conda``. To do so, simply run::

    pip install rsmtool


Note that if you are on macOS, you will need to have the following line in your ``.bashrc`` for RSMTool to work properly::

    export MPLBACKEND=Agg
