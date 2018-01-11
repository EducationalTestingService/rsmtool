RSMTool Release Process
=======================

This process is only meant for the project administrators, not users and developers.

1. Create a release branch on GitHub.

2. In that release branch, update the version numbers in ``version.py``, update the conda-recipe, and update the README, if necessary.

3. Build the new conda package locally on your mac using the following command::

    conda build -c defaults -c conda-forge --python=3.6 --numpy=1.13 rsmtool

4. Convert the package for both linux and windows::

    conda convert -p win-64 -p linux-64 <mac package tarball>

5. Upload all packages to anaconda.org using ``anaconda upload``.

6. Create a pull request on the `rsmtool-conda-tester <https://github.com/EducationalTestingService/rsmtool-conda-tester/>`_ repository to test the Linux and Windows packages.

7. Draft a release on GitHub while the Linux and Windows builds are running.

8. Once both builds have passed, make a pull request with the release branch to merge into ``master``.

9. Once the build for the PR passes, merge the branch into ``master``.

10. Make sure that the RTFD build for ``master`` passes.

11. Tag the latest commit in ``master`` with the appropriate release tag and publish the release on GitHub.

12. Do an accompanying release of RSMExtra (only needed for ETS users).

13. Update the RSMTool conda environment on the ETS linux servers with the latest packages for both RSMTool and RSMExtra.

14. Send an email around at ETS announcing the release and the changes.
