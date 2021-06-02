RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

#. Recreate the development environment so all unpinned packages are updated to their latest versions. See instructions for this `here <https://rsmtool.readthedocs.io/en/main/contributing.html#setting-up>`_.

#. Make sure any and all tests are passing in ``main``. Make sure you have also run tests locally in strict mode (``STRICT=1 nosetests --nologcapture tests``) to catch any warnings in the HTML report that can be fixed before the release.

#. Run the ``tests/update_files.py`` script with the appropriate arguments to make sure that all test data in the new release have correct experiment ids and filenames. If any (non-model) files need to be changed this should be investigated before the branch is released. Please see more details about running this `here <https://rsmtool.readthedocs.io/en/stable/contributing.html#writing-new-functional-tests>`_.

    .. note:: 

        Several files have been excluded from the repository due to their non-deterministic nature so please do not add them back to the repository. The following files are currently excluded:

            * Fairness test files for `lr-eval-system-score-constant` test
            * Predictions and all evaluation files for `linearsvr` test. 
     
        Note that the full set of outputs from these test files are also used as input for `rsmcompare` and `rsmsummarize` tests. These *input* files need to be updated following the process under **Example 2** in `Writing new functional tests <https://rsmtool.readthedocs.io/en/stable/contributing.html#writing-new-functional-tests>`_. You can also see `this pull request <https://github.com/EducationalTestingService/rsmtool/pull/525>`_ for more information. 

#. Create a release branch ``release/XX`` on GitHub.

#. In the release branch:

   #. update the version numbers in ``version.py``.

   #. update the conda recipe.

   #. update the documentation with any new features or details about changes.

   #. run ``make linkcheck`` on the documentation and fix any redirected/broken links.

   #. update the README and this release documentation, if necessary.

#. Build the PyPI source and wheel distributions using ``python setup.py sdist build`` and ``python setup.py bdist_wheel build`` respectively.

#. Upload the source and wheel distributions to TestPyPI  using ``twine upload --repository testpypi dist/*``. You will need to have the ``twine`` package installed and set up your ``$HOME/.pypirc`` correctly. See details `here <https://packaging.python.org/guides/using-testpypi/>`__.

#. Install the TestPyPI package as follows::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rsmtool

#. Then run some tests from a RSMTool working copy. If the TestPyPI package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

#. Build the new generic conda package by running the following command in the ``conda-recipe`` directory (note that this assumes that you have cloned RSMTool in a directory named ``rsmtool``)::

    conda build -c conda-forge -c ets .

#. This will create python 3.7, 3.8, and 3.9 packages for your native platform, e.g., ``osx-64``.

#. Convert these built packages for the other two platforms. For example, if you ran the above command on macOS, run ``conda convert -p linux-64 -p win-64 <packages files>``, where `<packages_files>` are the package files that were created in step 10.

#. Upload all 9 package files (3 Python versions x 3 platforms) to anaconda.org using ``anaconda upload --user ets <path_to_files>``. You will need to have the appropriate permissions for the ``ets`` organization. 

#. Create pull requests on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ and `rsmtool-pip-tester <https://github.com/EducationalTestingService/rsmtool-pip-tester/>`_ repositories to test the conda and TestPyPI packages on Linux and Windows.

#. Draft a release on GitHub while the Linux and Windows package tester builds are running.

#. Once both builds have passed, make a pull request with the release branch to be merged into ``main`` and request code review.

#. Once the build for the PR passes and the reviewers approve, merge the release branch into ``main``.

#. Upload source and wheel packages to PyPI using ``python setup.py sdist upload`` and ``python setup.py bdist_wheel upload``

#. Make sure that the ReadTheDocs build for ``main`` passes by examining the badge at this `URL <https://img.shields.io/readthedocs/rsmtool/latest>`_ - this should say "passing" in green.

#. Tag the latest commit in ``main`` with the appropriate release tag and publish the release on GitHub.

#. Make another PR to merge ``main`` branch into ``stable`` so that the the default ReadTheDocs build (which is ``stable``) always points to the latest release.

#. Update the CI plan for RSMExtra (only needed for ETS users) to use this newly built RSMTool conda package. Do any other requisite changes for RSMExtra. Once everything is done, do a release of RSMExtra.

#. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

#. Send an email around at ETS announcing the release and the changes.

#. Create a `Dash <https://kapeli.com/dash>`_ docset from the documentation by following the instructions :ref:`here <dash_docset>`.
