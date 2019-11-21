RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

1. Run the ``tests/update_files.py`` script with the appropriate arguments to make sure that all test data in the new release have correct experiment ids and filenames. If any (non-model) files need to be changed this should be investigated before the branch is released. 

2. Create a release branch on GitHub.

3. In that release branch, update the version numbers in ``version.py``, update the conda-recipe, and update the README, if necessary. You should also run `make linkcheck` on the documentation to fix and update any broken/redirected links.

4. Upload source and wheel packages to PyPI using ``python setup.py sdist upload`` and ``python setup.py bdist_wheel upload``

5. Build the new generic conda package locally on your mac using the following command::

    conda build -c conda-forge rsmtool

6. Upload the built package to anaconda.org using ``anaconda upload --user ets <package tarball>``.

7. Create pull requests on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ and `rsmtool-pip-tester <https://github.com/EducationalTestingService/rsmtool-pip-tester/>`_ repositories to test the conda and PyPI packages on Linux and Windows.

8. Draft a release on GitHub while the Linux and Windows builds are running.

9. Once both builds have passed, make a pull request with the release branch to be merged into ``master`` and request code review.

10. Once the build for the PR passes and the reviewers approve, merge the release branch into ``master``.

11. Make sure that the ReadTheDocs build for ``master`` passes.

12. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

13. Make another PR to merge ``master`` branch into ``stable`` so that the ``stable`` ReadTheDocs build always points to the latest release.

14. Update the CI plan for RSMExtra (only needed for ETS users) to use this newly built RSMTool conda package. Do any other requisite changes for RSMExtra. Once everything is done, do a release of RSMExtra.

15. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

16. Send an email around at ETS announcing the release and the changes.
