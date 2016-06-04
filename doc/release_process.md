0. Update the [changelog](CHANGELOG.md).
1. One final Circle CI build for `develop`.
2. Update version number in code.
3. Build and upload the conda package for Mac as follows:

    ```
    cd conda-recipe/unix
    conda build --python 3.4 rsmtool
    anaconda upload <mac package tarball>
    ```

4. Convert the package for linux and upload it:

    ```
    conda convert -p linux-64 <mac package tarball>
    anaconda upload linux-64/<tarball>
    ```

5.  Build the conda package using the `windows` recipe on Mac/Linux, then convert it for the Windows platform, and then upload it:

    ```
    cd conda-recipe/windows
    conda build --python 3.4 rsmtool
    conda convert -p win-64 <package tarball>
    anaconda upload win-64/<tarball>
    ```

6. Test all three conda packages in fresh conda environments.
7. Install conda package on ETS linux servers in the python 3 environment.
8. Merge `develop` into `master`. 
9. Tag the latest commit to master with the appropriate release tag.

