RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

1. Run `tests/update_files.py` to make sure that all test data in the new release have correct experiment ids and filenames. If any (non-model) files need to be changed this should be investigated before the branch is released. 

2. Create a release branch on GitHub.

3. In that release branch, update the version numbers in ``version.py``, update the conda-recipe, and update the README, if necessary.

4. Build the new conda package locally on your mac using the following command::

    conda build -c defaults -c conda-forge --python=3.6 --numpy=1.13 rsmtool

5. Convert the package for both linux and windows::

    conda convert -p win-64 -p linux-64 <mac package tarball>

6. Upload all packages to anaconda.org using ``anaconda upload``.

7. Upload source package to PyPI using ``python setup.py sdist upload``.

8. Create pull requests on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ and `rsmtool-pip-tester <https://github.com/EducationalTestingService/rsmtool-pip-tester/>`_ repositories to test the conda and PyPI packages on Linux and Windows.

9. Draft a release on GitHub while the Linux and Windows builds are running.

10. Once both builds have passed, make a pull request with the release branch to be merged into ``master`` and request code review.

11. Once the build for the PR passes and the reviewers approve, merge the release branch into ``master``.

12. Make sure that the RTFD build for ``master`` passes.

13. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

14. Do an accompanying release of RSMExtra (only needed for ETS users).

15. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

16. Send an email around at ETS announcing the release and the changes.
