0. Update the [changelog](CHANGELOG.md).
1. One final Travis CI build for `develop`.
2. Update version number in code.
3. Build the conda package for Mac as follows:

    ```
    cd conda-recipe
    conda build --python 3.4 rsmtool
    ```

4. Convert the package for both linux and windows:

    ```
    conda convert -p win-64 -p linux-64 <mac package tarball>
    ```

5. Test all three conda packages in fresh conda environments.
6. Install conda package on ETS linux servers in the python 3 environment.
7. Merge `develop` into `master`. 
8. Tag the latest commit to master with the appropriate release tag.

