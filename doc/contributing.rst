Contributing to RSMTool
=======================

Contributions to RSMTool are very welcome. You can use the instructions below to get started on developing new features or functionality for RSMTool. When contributing to RSMTool, all of your contributions must come with updates to documentations as well as tests. 

Setting up
----------

To set up a local development environment, follow the steps below:

1. Pull the latest version of RSMTool from GitHub and switch to the ``master`` branch.

2. If you already have the ``conda`` package manager installed, skip to the next step. If you do not, follow the instructions on `this page <https://conda.io/docs/user-guide/install/index.html>`_ to install conda.

3. Create a new conda environment (say, ``rsmtool``) and install the packages specified in the ``conda_requirements.txt`` file by running::

    conda create -n rsmtool -c defaults -c conda-forge -c desilinguist --file conda_requirements.txt

4. Activate the environment using ``source activate rsmtool`` (use ``activate rsmtool`` if you are on Windows).

5. Run ``pip install -e .`` to install rsmtool into the environment in editable mode which is what we need for development.

6. Create a new git branch with a useful and descriptive name.

7. Make your changes and add tests. See the next section for more on writing new tests. 

8. Run ``nosetests -v --nologcapture tests`` to run the tests. We use the ``--nologcapture`` switch, since otherwise test failures for some tests tend to produce very long Jupyter notebook traces.

RSMTool tests
-------------

Existing tests for RSMTool are spread across the various ``test_*.py`` files under the ``tests`` directory after you check out the RSMTool source code from GitHub. 

There are two kinds of existing tests in RSMTool: 

1. The first type of tests are **unit tests**, i.e., very specific tests for which you have a single example (usually embedded in the test itself) and you compare the generated output with known or expected output. These tests should have a very narrow and well defined scope. To see examples of such unit tests, see the test functions in the file `tests/test_utils.py`. 

2. The second type of tests are **functional tests** which are generally written from the users' perspective to test that RSMTool is doing things that users would expect it to. In RSMTool, most (if not all) functional tests are written in the form of "experiment tests", i.e., we first define an experimental configuration using an RSMTool (or RSMEval/RSMPredict, RSMCompare, RSMSummarize) configuration file, then we run the experiment, and then compare the generated output files to expected output files to make sure that RSMTool components are operating as expected. To see examples of such tests, you can look at any of the ``tests/test_experiment_*.py`` files. 

.. note:: 

    RSMTool functional tests are *parameterized*, i.e., since most are identical other than the configuration file that needs to be run, the basic functionality of the test has been factored out into utility functions. Each line starting with `param` in any of the ``test_experiment_*.py`` files represents a specific functional test.

Any new contributions to RSMTool, no matter how small or trivial, *must* be accompanied by updates to documentations as well as new unit and/or functional tests. Adding new unit tests is fairly straightforward. However, adding new functional tests is a little more involved. 

Writing new functional tests
----------------------------

To write a new experiment test for RSMTool (or any of the other tools):

    (a) Create a new directory under ``tests/data/experiments`` using a descriptive name. 

    (b) Create a JSON configuration file under that directory with the various fields appropriately set for what you want to test. Feel free to use multiple words separated by hyphens to come up with a name that describes the testing condition. The name of the configuration file should be the same as the value of the ``experiment_id`` field in your JSON file. By convention, that's usually the same as the name of the directory you created but with underscores instead of hyphens. 

    (c) Next, you need to add the test to the list of parameterized tests in the appropriate test file based on the tool for which you are adding the test, e.g., RSMEval tests should be added to ``tests/test_experiment_rsmeval.py``, RSMPredict tests to ``tests/test_experiment_rsmpredict.py``, and so on. RSMTool tests can be added to any of the four files. The arguments for the `param()` call can be found in the :ref:`Table 1 <param_table>` below.

    (d) In some rare cases, you might want to use a non-parameterized experiment test if you are doing something very different. These should be few and far between. Examples of these can also be seen in various ``tests/test_experiment_*.py`` files.

    (e) Another rare scenario is the need to create an entirely new ``tests/test_experiment_X.py`` file instead of using one of the existing ones. This should *not* be necessary unless you are trying to test a newly added tool or component. 

    .. _param_table:
    .. table:: Table 1: Arguments for ``param()`` when adding new parameterized functional tests
        :widths: auto

        +----------------------------------------------------------------------------+
        | Writing test(s) for RSMTool                                                |
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
        | Writing test(s) for RSMEval                                                |
        |                                                                            |
        | * Same arguments as RSMTool except the ``skll`` keyword argument is not    |
        |   applicable.                                                              |
        +----------------------------------------------------------------------------+
        | Writing test(s) for RSMPredict                                             |
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
        | Writing test(s) for RSMCompare                                             |
        |                                                                            |
        | * First positional argument is the name of the test directory you created. |
        |                                                                            |
        | * Second positional argument is the comparison ID from the JSON            |
        |   configuration file.                                                      |
        +----------------------------------------------------------------------------+
        | Writing test(s) for RSMSummarize                                           |
        |                                                                            |
        | * The only positional argument is the name of the test directory you       |
        |   created.                                                                 |
        |                                                                            |
        | * Set ``file_format="tsv"`` (or ``"xlsx"``) if you specified the same      |
        |   field in the configuration file.                                         |
        +----------------------------------------------------------------------------+

Once you have added all new functional tests, commit all of your changes. Next, you should run ``nosetests --nologcapture`` to run all the tests. Obviously, the newly added tests will fail since you have not yet generated the expected output for that test. 

To do this, you should now run the following:

.. _update_files:
.. code-block:: text
    
    python tests/update_files.py --tests tests --outputs test_outputs

This will copy over the generated outputs for the newly added tests and show you a report of the files that it added. If run correctly, the report should *only* refer to model files (``*.model``/``*.ols``) and the files affected by the functionality you implemented. If you run ``nosetests`` again, your newly added tests should now pass. 

At this point, you should inspect all of the new test files added by the above command using to make sure that the outputs are as expected. You can find these files under ``tests/data/experiments/<test>/output`` where ``<test>`` refers to the test(s) that you added. Once you are satisified that the outputs are as expected, you can commit all the them.

Advanced tips and tricks
------------------------

Here are some advanced tips and tricks when working with RSMTool tests.

1. To run a specific test function in a specific test file, simply use ``nosetests --nologcapture tests/test_X.py:Y`` where ``test_X.py`` is the name of the test file, and ``Y`` is the test functions. Note that this will not work for parameterized tests. If you want to run a specific parameterized test, you can comment out all of the other ``param()`` calls and run the ``test_run_experiment_parameterized()`` function as above.

2. If you make any changes to the code that can change the output that the tests are expected to produce, you *must* re-run all of the tests and then update the *expected* test outputs using the ``update_files.py`` command as shown :ref:`above <update_files>`.

3. In the rare case that you *do* need to create an entirely new ``tests/test_experiment_X.py`` file instead of using one of the existing ones, you can choose whether to exclude the tests contained in this file from updating their expected outputs when ``update_files.py`` is run by setting ``_AUTO_UPDATE=False`` at the top of the file. This should *only* be necessary if you are absolutely sure that your tests will never need updating.

