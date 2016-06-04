1. Update the [changelog](CHANGELOG.md).2
2. One final Circle CI build for `master`.
3. Make a release branch.
4. Update version number in code.
5. Make any other changes e.g., to conda-recipe and conda requirements files if the dependencies have changed.
6. Build and upload the conda package for Mac as follows:

    ```
    cd conda-recipe/unix
    conda build --python 3.4 rsmtool
    anaconda upload <mac package tarball>
    ```

7. Convert the package for linux and upload it:

    ```
    conda convert -p linux-64 <mac package tarball>
    anaconda upload linux-64/<tarball>
    ```

8.  Build the conda package using the `windows` recipe on Mac/Linux, then convert it for the Windows platform, and then upload it:

    ```
    cd conda-recipe/windows
    conda build --python 3.4 rsmtool
    conda convert -p win-64 <package tarball>
    anaconda upload win-64/<tarball>
    ```

9. Test all three conda packages in fresh conda environments.
10. Install conda package on ETS linux servers in the python 3 environment.
11. Merge `master` into `stable`. 
12. Tag the latest commit to `stable` with the appropriate release tag.

