.. _automated_configuration:

Auto-generating configuration files
-----------------------------------
Configuration files for :ref:`rsmtool <config_file_rsmtool>`, :ref:`rsmeval <config_file_rsmeval>`, :ref:`rsmcompare <config_file_rsmcompare>`, :ref:`rsmpredict <config_file_rsmpredict>`, and :ref:`rsmsummarize <config_file_rsmsummarize>` can be difficult to create manually due to the large number of configuration options supported by these tools. To make this easier for users, all of these tools support *automatic* creation of configuration files, both interactively and non-interactively.

.. note:: Although we only use ``rsmtool`` in the examples below, the same instructions apply to all 5 tools; simply replace ``rsmtool`` with ``rsmeval``, ``rsmcompare``, etc.

Interactive generation
~~~~~~~~~~~~~~~~~~~~~~
For novice users, it is easiest to use a guided, interactive mode to generate a configuration file.
For example, to generate an ``rsmtool`` configuration file interactively, run the following command:

.. code-block:: bash

    rsmtool generate --interactive > example_rsmtool.json

The following animated image shows an example interactive session after the above command is run (click to play):

.. raw:: html
   
   <script id="asciicast-313107" src="https://asciinema.org/a/313107.js" data-autoplay="false" async></script>


The configuration file ``example_rsmtool.json`` generated via the session is shown below:

.. literalinclude:: assets/example_rsmtool.json
    :language: javascript

The following are important things to note about interactive generation:

- Carefully read the instructions and notes displayed at the top when you first enter interactive mode.

- If you do not redirect the output of the command to a file, the generated configuration file will simply be printed out.

- You may see messages like "invalid option", "invalid file" on the bottom left while you are entering the value for a field. This is an artifact of real-time validation. For example, when choosing a training file for ``rsmtool``, the message "invalid file" may be displayed while you navigate to the actual file. Once you get to a valid file, this message should disappear.

- Required fields will *not* accept a blank input (just pressing enter) and show an error message in the bottom left until a valid input is provided.

- Optional fields will accept blank inputs since they have default values that will be used if no user input is provided. In some cases, default values are shown underlined in parentheses. 

- The configuration files generated interactively contain comments (as indicated by ``// ...``). While RSMTool handles JSON files with comments just fine, you may need to remove the comments manually if you wish to use these files outside of RSMTool.
