RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

1. Make sure any and all tests are passing in ``master``. Make sure you have also run tests locally in strict mode (``STRICT=1 nosetests --nologcapture tests``) to catch any deprecation warnings in the HTML report that can be fixed before the release.

2. Run the ``tests/update_files.py`` script with the appropriate arguments to make sure that all test data in the new release have correct experiment ids and filenames. If any (non-model) files need to be changed this should be investigated before the branch is released. 

3. Create a release branch ``release/XX`` on GitHub.

4. In the release branch:

   a. update the version numbers in ``version.py``.

   b. update the conda recipe.

   c. update the documentation with any new features or details about changes.

   d. run ``make linkcheck`` on the documentation and fix any redirected/broken links.

   e. update the README and this release documentation, if necessary.

5. Build the PyPI source and wheel distributions using ``python setup.py sdist build`` and ``python setup.py bdist_wheel build`` respectively.

6. Upload the source and wheel distributions to TestPyPI  using ``twine upload --repository testpypi dist/*``. You will need to have the ``twine`` package installed and set up your ``$HOME/.pypirc`` correctly. See details `here <https://packaging.python.org/guides/using-testpypi/>`__.

7. Install the TestPyPI package as follows::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rsmtool

8. Then run some tests from a RSMTool working copy. If the TestPyPI package works, then move on to the next step. If it doesn't, figure out why and rebuild and re-upload the package.

9. Build the new generic conda package by running the following command in the ``conda-recipe`` directory (note that this assumes that you have cloned RSMTool in a directory named ``rsmtool`` and that the latest version of ``numpy`` is ``1.18``)::

    conda build -c conda-forge -c ets --numpy=1.18 .

10. Upload the package to anaconda.org using ``anaconda upload --user ets <package tarball>``. You will need to have the appropriate permissions for the ``ets`` organization. 

11. Create pull requests on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ and `rsmtool-pip-tester <https://github.com/EducationalTestingService/rsmtool-pip-tester/>`_ repositories to test the conda and TestPyPI packages on Linux and Windows.

12. Draft a release on GitHub while the Linux and Windows package tester builds are running.

13. Once both builds have passed, make a pull request with the release branch to be merged into ``master`` and request code review.

14. Once the build for the PR passes and the reviewers approve, merge the release branch into ``master``.

15. Upload source and wheel packages to PyPI using ``python setup.py sdist upload`` and ``python setup.py bdist_wheel upload``

16. Make sure that the ReadTheDocs build for ``master`` passes by examining the badge at this `URL <https://img.shields.io/readthedocs/rsmtool/latest>`_ - this should say "passing" in green.

17. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

18. Make another PR to merge ``master`` branch into ``stable`` so that the the default ReadTheDocs build (which is ``stable``) always points to the latest release.

19. Update the CI plan for RSMExtra (only needed for ETS users) to use this newly built RSMTool conda package. Do any other requisite changes for RSMExtra. Once everything is done, do a release of RSMExtra.

20. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

21. Send an email around at ETS announcing the release and the changes.

22. Create a `Dash <https://kapeli.com/dash>`_ docset from the documentation by following the instructions :ref:`here <dash_docset>`.

