.. _install:

Installation
============
Note that RSMTool currently only works with Python 3.6.x.

Installing with conda
----------------------

Currently, the recommended way to install RSMTool is by using the ``conda`` package manager. If you have already installed ``conda``, you can skip straight to Step 2.

1. To install ``conda``, follow the instructions on `this page <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. 

2. Create a new conda environment (say, ``rsmtool``) and install the RSMTool conda package for your preferred Python version. For example, for Python 3.8, run:::

    conda create -n rsmtool -c conda-forge -c ets python=3.6 rsmtool

3. Activate this conda environment by running ``conda activate rsmtool``. You should now have all of the RSMTool command-line utilities in your path. [#]_

4. From now on, you will need to activate this conda environment whenever you want to use RSMTool. This will ensure that the packages required by RSMTool will not affect other projects.

RSMTool can also be downloaded directly from
`GitHub <https://github.com/EducationalTestingService/rsmtool>`_.

Installing with pip
-------------------

You can also use ``pip`` to install RSMTool instead of ``conda``. To do so, simply run::

    pip install rsmtool

Note that if you are on macOS, you will need to have the following line in your ``.bashrc`` for RSMTool to work properly::

    export MPLBACKEND=Agg


.. rubric:: Footnotes

.. [#] For older versions of conda, you may have to do ``source activate rsmtool`` on Linux/macOS and ``activate rsmtool`` on Windows.
