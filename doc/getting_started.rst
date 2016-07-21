.. _install:

Installation
============
Note that RSMTool only currently works with Python 3.4 and higher.

Currently, the best way to install RSMTool is by using the ``conda`` package manager. If you have already installed ``conda``, you can skip straight to Step 2.

1. To install ``conda``, follow the instructions on `this page <http://conda.pydata.org/docs/install/quick.html>`_.

2. Create a new conda environment (say, ``rsmtool``) and install the RSMTool conda package by running::

    conda create -n rsmtool -c desilinguist python=3.4 rsmtool

3. Activate this conda environment by running ``source activate rsmtool`` (``activate rsmtool`` on windows). You should now have all of the RSMTool command-line utilities in your path.

4. From now on, you will need to activate this conda environment whenever you want to use RSMTool. This will ensure that the packages required by RSMTool will not affect other projects.

RSMTool can also be downloaded directly from
`GitHub <http://github.com/EducationalTestingService/rsmtool>`_.
