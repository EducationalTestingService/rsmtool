.. _dash_docset:


Generating Dash Docset
=======================

These instructions are for creating a docset based on the RSMTool documentation to be
used for the macOS `Dash <https://kapeli.com/dash>`_ app.

1. If you have created/deleted any documentation files, update the variable ``FILES_TO_MODIFY`` at the top of the file ``doc/add_dash_anchors.py`` to reflect these changes.

2. Run ``make dash`` in the ``doc`` directory. This will compile the HTML documentation using the Alabaster theme (which is better suited for this purpose), run ``add_dash_anchors.py`` for Dash TOC support, and create the docset file (``_build/RSMTool.docset``).

3. Clone the repository at https://github.com/Kapeli/Dash-User-Contributions.

4. Follow the instructions in that repository's README. Note that we do not need add explicit icons for the docset since our icon is already included in the docset file created above. Make sure to submit a pull request to that repo in the end!
