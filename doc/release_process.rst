RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

1. Run `update_test_files.py` to make sure that all test data in the new release have correct experiment ids and filenames. If any files need to be changed this should be investigated before the branch is released. 

2. Create a release branch on GitHub.

3. In that release branch, update the version numbers in ``version.py``, update the conda-recipe, and update the README, if necessary.

4. Build the new conda package locally on your mac using the following command::

    conda build -c defaults -c conda-forge --python=3.6 --numpy=1.13 rsmtool

5. Convert the package for both linux and windows::

    conda convert -p win-64 -p linux-64 <mac package tarball>

6. Upload all packages to anaconda.org using ``anaconda upload``.

7. Create a pull request on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ repository to test the Linux and Windows packages.

8. Draft a release on GitHub while the Linux and Windows builds are running.

9. Once both builds have passed, make a pull request with the release branch to be merged into ``master`` and request code review.

10. Once the build for the PR passes and the reviewers approve, merge the release branch into ``master``.

11. Make sure that the RTFD build for ``master`` passes.

12. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

13. Do an accompanying release of RSMExtra (only needed for ETS users).

14. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

15. Send an email around at ETS announcing the release and the changes.