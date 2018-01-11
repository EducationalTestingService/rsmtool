Contributing to RSMTool
=======================

Contributions to RSMTool are very welcome. You can use the instructions below to get started on developing new features or functionality for RSMTool.

1. Pull the latest version of RSMTool from GitHub and switch to the ``master`` branch.

2. If you already have the ``conda`` package manager installed, skip to the next step. If you do not, follow the instructions on `this page <http://conda.pydata.org/docs/install/quick.html>`_ to install conda.

3. Create a new conda environment (say, ``rsmtool``) and install the packages specified in the ``conda_requirements.txt`` file by running::

    conda create -n rsmtool -c defaults -c conda-forge -c desilinguist --file conda_requirements.txt

4. Activate the environment using ``source activate rsmtool`` (use ``activate rsmtool`` if you are on Windows).

5. Run ``pip install -e .`` to install rsmtool into the environment in editable mode which is what we need for development.

6. Run ``nosetests -v tests`` to run the tests.
