Contributing to RSMTool
=======================

Contributions to RSMTool are very welcome. You can use the instructions below to get started on developing new features or functionality for RSMTool. When contributing to RSMTool, all of your contributions must come with updates to documentation as well as tests.

Setting up
----------

To set up a local development environment, follow the steps below:

#. Clone the `Github repository <https://github.com/EducationalTestingService/rsmtool>`_ for RSMTool. If you have already have a local version of the repository, pull the latest version from GitHub and switch to the ``main`` branch.

#. If you already have the ``conda`` package manager installed, skip to the next step. If you do not, follow the instructions on `this page <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ to install conda.

#. Create a new conda environment (say, ``rsmdev``) and install the packages specified in the ``requirements.dev`` file by running::

    conda create -n rsmdev -c conda-forge -c ets --file requirements.dev

#. Activate the environment using ``conda activate rsmdev``.

#. Run ``pip install -e .`` to install rsmtool into the environment in editable mode which is what we need for development.

#. Run ``pre-commit install --hook-type pre-commit --hook-type commit-msg --overwrite --install-hooks`` to install the git hooks for pre-commit checks.

#. Create a new git branch with a useful and descriptive name.

#. Make your changes and add tests. See the next section for more on writing new tests.

#. Run ``nose2 --quiet -s tests`` to run the tests. We use the ``--quiet`` switch, since otherwise test failures for some tests tend to produce very long Jupyter notebook traces.

Documentation
-------------

Note that the file ``doc/requirements.txt`` is meant specifically for the ReadTheDocs documentation build process and should not be used locally. To build the documentation locally, you *must* use the same conda environment created above.

Code style
----------
The RSMTool codebase enforces a certain code style via pre-commit checks and this style is automatically applied to any new contributions. This code style consists of:

#. All Python code is formatted via the `black <https://black.readthedocs.io/en/stable/>`_ pre-commit check.

#. The f-string specification is used for all format strings in the Python code as enforced by the `flynt <https://pypi.org/project/flynt/>`_ pre-commit check.

#. Any code violations for `PEP 8 <https://peps.python.org/pep-0008/>`_, `PEP 257 <https://peps.python.org/pep-0257/>`_, and import statements are checked using the `ruff <https://github.com/astral-sh/ruff-pre-commit>`_ pre-commit check.

#. Commit messages must follow the `conventional commit specification <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_. This is enforced by the `conventional-pre-commit <https://github.com/compilerla/conventional-pre-commit>`_ pre-commit check.

#. The imports at the top of any Python files are grouped and sorted as follows: STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER. As an example, consider the imports at the top of ``reporter.py`` which look like this:

    .. code-block:: python

        import argparse
        import asyncio
        import json
        import logging
        import os
        import sys
        from os.path import abspath, basename, dirname, join, splitext

        from nbconvert.exporters import HTMLExporter
        from nbconvert.exporters.templateexporter import default_filters
        from traitlets.config import Config

        from .reader import DataReader

    Rather than doing this grouping and sorting manually, we use the `isort <https://pycqa.github.io/isort/>`_ pre-commit hook to achieve this.

#. All classes, functions, and methods in the main code files have `numpy-formatted docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_ that comply with `PEP 257 <https://peps.python.org/pep-0257/>`_. This is enforced via the `pydocstyle <http://www.pydocstyle.org/en/stable/>`_ pre-commit check. Additionally, when writing docstrings, make sure to use the appropriate quotes when referring to argument names vs. argument values. As an example, consider the docstring for the `train_skll_model <https://rsmtool.readthedocs.io/en/main/api.html#rsmtool.modeler.Modeler.train_skll_model>`_  method of the ``rsmtool.modeler.Modeler`` class. Note that string argument values are enclosed in double quotes (e.g., "csv", "neg_mean_squared_error") whereas values of other built-in types are written as literals (e.g., ``True``, ``False``, ``None``). Note also that if one had to refer to an argument name in the docstring, this referent should be written as a literal. In general, we strongly encourage looking at the docstrings in the existing code to make sure that new docstrings follow the same practices.

RSMTool tests
-------------

Existing tests for RSMTool are spread across the various ``test_*.py`` files under the ``tests`` directory after you check out the RSMTool source code from GitHub.

There are two kinds of existing tests in RSMTool:

#. The first type of tests are **unit tests**, i.e., very specific tests for which you have a single example (usually embedded in the test itself) and you compare the generated output with known or expected output. These tests should have a very narrow and well defined scope. To see examples of such unit tests, see the test functions in the file `tests/test_utils.py`.

#. The second type of tests are **functional tests** which are generally written from the users' perspective to test that RSMTool is doing things that users would expect it to. In RSMTool, most (if not all) functional tests are written in the form of "experiment tests", i.e., we first define an experimental configuration using an ``rsmtool`` (or ``rsmeval``/``rsmpredict``/``rsmcompare``/``rsmsummarize``) configuration file, then we run the experiment, and then compare the generated output files to expected output files to make sure that RSMTool components are operating as expected. To see examples of such tests, you can look at any of the ``tests/test_experiment_*.py`` files.

.. note::

    RSMTool functional tests are *parameterized*, i.e., since most are identical other than the configuration file that needs to be run, the basic functionality of the test has been factored out into utility functions. Each line starting with `param` in any of the ``test_experiment_*.py`` files represents a specific functional test.

Any new contributions to RSMTool, no matter how small or trivial, *must* be accompanied by updates to documentations as well as new unit and/or functional tests. Adding new unit tests is fairly straightforward. However, adding new functional tests is a little more involved.

Writing new functional tests
----------------------------

To write a new experiment test for RSMTool (or any of the other tools):

    (a) Create a new directory under ``tests/data/experiments`` using a descriptive name.

    (b) Create a JSON configuration file under that directory with the various fields appropriately set for what you want to test. Feel free to use multiple words separated by hyphens to come up with a name that describes the testing condition. The name of the configuration file should be the same as the value of the ``experiment_id`` field in your JSON file. By convention, that's usually the same as the name of the directory you created but with underscores instead of hyphens. If you are creating a new test for ``rsmcompare`` or ``rsmsummarize``, copy over one or more of the existing ``rsmtool`` or ``rsmeval`` test experiments as input(s) and keep the same name. This will ensure that these inputs will be regularly updated and remain consistent with the current outputs generated by these tools. If you must create a test for a scenario not covered by a current tool, create a new ``rsmtool``/``rsmeval`` test first following the instructions on this page.

    (c) Next, you need to add the test to the list of parameterized tests in the appropriate test file based on the tool for which you are adding the test, e.g., ``rsmeval`` tests should be added to ``tests/test_experiment_rsmeval.py``, ``rsmpredict`` tests to ``tests/test_experiment_rsmpredict.py``, and so on. Tests for ``rsmtool`` can be added to any of the four files. The arguments for the `param()` call can be found in the :ref:`Table 1 <param_table>` below.

    (d) In some rare cases, you might want to use a non-parameterized experiment test if you are doing something very different. These should be few and far between. Examples of these can also be seen in various ``tests/test_experiment_*.py`` files.

    (e) Another rare scenario is the need to create an entirely new ``tests/test_experiment_X.py`` file instead of using one of the existing ones. This should *not* be necessary unless you are trying to test a newly added tool or component.

    .. _param_table:
    .. table:: Table 1: Arguments for ``param()`` when adding new parameterized functional tests
        :widths: auto

        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmtool``                                            |
        |                                                                            |
        | * First positional argument is the name of the test directory you created. |
        |                                                                            |
        | * Second positional argument is the experiment ID from the JSON            |
        |   configuration file.                                                      |
        |                                                                            |
        | * Use ``consistency=True`` if you have set `second_human_score_column` in  |
        |   the configuration file.                                                  |
        |                                                                            |
        | * Use ``skll=True`` if you are writing a test for a SKLL model.            |
        |                                                                            |
        | * Set ``subgroups`` keyword argument to the same list that you specified   |
        |   in the configuration file.                                               |
        |                                                                            |
        | * Set ``file_format="tsv"`` (or ``"xlsx"``) if you specified the same      |
        |   field in the configuration file.                                         |
        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmeval``                                            |
        |                                                                            |
        | * Same arguments as RSMTool except the ``skll`` keyword argument is not    |
        |   applicable.                                                              |
        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmpredict``                                         |
        |                                                                            |
        | * The only positional argument is the name of the test directory you       |
        |   created.                                                                 |
        |                                                                            |
        | * Use ``excluded=True`` if you want to check the excluded responses file   |
        |   as part of the test.                                                     |
        |                                                                            |
        | * Set ``file_format="tsv"`` (or ``"xlsx"``) if you specified the same      |
        |   field in the configuration file.                                         |
        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmcompare``                                         |
        |                                                                            |
        | * First positional argument is the name of the test directory you created. |
        |                                                                            |
        | * Second positional argument is the comparison ID from the JSON            |
        |   configuration file.                                                      |
        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmsummarize``                                       |
        |                                                                            |
        | * The only positional argument is the name of the test directory you       |
        |   created.                                                                 |
        |                                                                            |
        | * Set ``file_format="tsv"`` (or ``"xlsx"``) if you specified the same      |
        |   field in the configuration file.                                         |
        +----------------------------------------------------------------------------+
        | Writing test(s) for ``rsmexplain``                                         |
        |                                                                            |
        | * First positional argument is the name of the test directory you created. |
        |                                                                            |
        | * Second positional argument is the experiment ID from the JSON            |
        |   configuration file.                                                      |
        +----------------------------------------------------------------------------+

Once you have added all new functional tests, commit all of your changes. Next, you should run ``nose2`` to run all the tests. Obviously, the newly added tests will fail since you have not yet generated the expected output for that test.

To do this, you should now run the following:

.. _update_files:
.. code-block:: text

    python tests/update_files.py --tests tests --outputs test_outputs

This will copy over the generated outputs for the newly added tests and show you a report of the files that it added. It will also update the input files form tests for ``rsmcompare`` and ``rsmsummarize``. If run correctly, the report should *only* refer to the files affected by the functionality you implemented. If you run ``nose2`` again, your newly added tests should now pass.

At this point, you should inspect all of the new test files added by the above command to make sure that the outputs are as expected. You can find these files under ``tests/data/experiments/<test>/output`` where ``<test>`` refers to the test(s) that you added.

However, if your changes resulted in updates to the inputs to ``rsmsummarize`` or ``rsmcompare`` tests, you will first need to re-run the tests for these two tools and then re-run the ``update_files.py`` to update the outputs.

Once you are satisified that the outputs are as expected, you can commit them.

The two examples below might help make this process easier to understand:

.. topic:: Example 1: You made a code change to better handle an edge case that only affects one test.

    #. Run ``nose2 --quiet -s tests``. The affected test failed.

    #. Run ``python tests/update_files.py --tests tests --outputs test_outputs`` to update test outputs. You will see the total number of deleted, updated and missing files. There should be no deleted files and no missing files. Only the files for your new test should be updated. There are no warnings in the output.

    #. If this is the case, you are now ready to commit your change and the updated test outputs.

.. topic:: Example 2: You made a code change that changes the output of many tests. For example, you renamed one of the evaluation metrics.

     #. Run ``nose2 --quiet -s tests``. Many tests will now fail since the output produced by the tool(s) has changed.

     #. Run ``python tests/update_files.py --tests tests --outputs test_outputs`` to update test outputs. The files affected by your change are shown as added/deleted. You also see the following warning:

        .. code-block:: text

            WARNING: X input files for rsmcompare/rsmsummarize tests have been updated. You need to re-run these tests and update test outputs

     #. This means that the changes you made to the code changed the outputs for one or more ``rsmtool``/``rsmeval`` tests that served as inputs to one or more ``rsmcompare``/``rsmsummarize`` tests. Therefore, it is likely that the current test outputs no longer match the expected output and the tests for those two tools must be be re-run.

     #. Run ``nose2 --quiet -s tests $(find tests -name 'test_*rsmsummarize*.py' | cut -d'/' -f2 | sed 's/.py//')`` and ``nose2 --quiet -s tests $(find tests -name 'test_*rsmcompare*.py' | cut -d'/' -f2 | sed 's/.py//')``. If you see any failures, make sure they are related to the changes you made since those are expected.

     #. Next, re-run ``python tests/update_files.py --tests tests --outputs test_outputs`` which should only update the outputs for the ``rsmcompare``/``rsmsummarize`` tests.

     #. If this is the case, you are now ready to commit your changes.


Advanced tips and tricks
------------------------

Here are some advanced tips and tricks when working with RSMTool tests.

#. To run a specific test function in a specific test file, simply use ``nose2 --quiet -s tests test_X.Y.Z`` where ``test_X`` is the name of the test file, ``Y`` is the enclosing ``unittest.TestCase`` subclass, and ``Z`` is the desired test function. Note that this will not work for parameterized tests. If you want to run a specific parameterized test, you can comment out all of the other parameters in the ``params`` and run the ``test_run_experiment_parameterized()`` function as above.

#. If you make any changes to the code that can change the output that the tests are expected to produce, you *must* re-run all of the tests and then update the *expected* test outputs using the ``update_files.py`` command as shown :ref:`above <update_files>`.

#. In the rare case that you *do* need to create an entirely new ``tests/test_experiment_X.py`` file instead of using one of the existing ones, you can choose whether to exclude the tests contained in this file from updating their expected outputs when ``update_files.py`` is run by setting ``_AUTO_UPDATE=False`` at the top of the file. This should *only* be necessary if you are absolutely sure that your tests will never need updating.

#. The ``--debugger/-D`` option for ``nose2`` is your friend. If you encounter test errors or test failures where the cause may not be immediately clear, re-run the ``nose2`` command with this option. Doing so will drop you into an interactive PDB session as soon as an error (or failure) is encountered and then you inspect the variables at that point (or use "u" and "d" to go up and down the call stack). This may be particularly useful for tests in ``tests/test_cli.py`` that use ``subprocess.run()``. If these tests are erroring out, use ``-D`` and inspect the "stderr" variable in the resulting PDB session to see what the error is.

#. In RSMTool 8.0.1 and later, the tests will pass even if any of the reports contain warnings. To catch any warnings that may appear in the reports, run the tests in strict mode (``STRICT=1 nose2 --quiet -s tests``).
