.. _autogen_configuration:

Auto-generating configuration files
-----------------------------------
Configuration files for :ref:`rsmtool <config_file_rsmtool>`, :ref:`rsmeval <config_file_rsmeval>`, :ref:`rsmcompare <config_file_rsmcompare>`, :ref:`rsmpredict <config_file_rsmpredict>`, and :ref:`rsmsummarize <config_file_rsmsummarize>` can be difficult to create manually due to the large number of configuration options supported by these tools. To make this easier for users, all of these tools support *automatic* creation of configuration files, both interactively and non-interactively.

Interactive generation
~~~~~~~~~~~~~~~~~~~~~~
For novice users, it is easiest to use a guided, interactive mode to generate a configuration file.
For example, to generate an ``rsmtool`` configuration file interactively, run the following command:

.. code-block:: bash

    rsmtool generate --interactive > example_rsmtool.json

The following screencast shows an example interactive session after the above command is run (click to play):

.. raw:: html
   
   <script id="asciicast-313107" src="https://asciinema.org/a/313107.js" data-autoplay="false" async></script>


The configuration file ``example_rsmtool.json`` generated via the session is shown below:

.. literalinclude:: assets/example_rsmtool.json
    :language: javascript

.. note:: Although we use ``rsmtool`` in the example above, the same instructions apply to all 5 tools; simply replace ``rsmtool`` with ``rsmeval``, ``rsmcompare``, etc.

There are some configuration options that can accept multiple inputs. For example, the ``experiment_dirs`` option for ``rsmsummarize`` takes a list of ``rsmtool`` experiment directories for a summary report. These options are handled differently in interactive mode. To illustrate this, let's generate a configuration file for ``rsmsummarize`` by using the following command:

.. code-block:: bash

    rsmsummarize generate --interactive > example_rsmsummarize.json

The following screencast shows the interactive session (click to play):

.. raw:: html
    
   <script id="asciicast-313149" src="https://asciinema.org/a/313149.js" data-autoplay="false" async></script>

And here is the generated configuration file for ``rsmsummarize``:

.. literalinclude:: assets/example_rsmsummarize.json
    :language: javascript

.. important:: 

    If you want to include subgroup information in the reports for ``rsmtool``, ``rsmeval``, and ``rsmcompare``, you should add ``--subgroups`` to the command. For example, when you run ``rsmeval generate --interactive --subgroups`` you would be prompted to enter the subgroup column names and the ``general_sections`` list will also include subgroup-based sections. Since the ``subgroups`` option can accept multiple inputs, it is handled in the same way as the ``experiment_dirs`` option for ``rsmsummarize`` above. 

We end with a list of important things to note about interactive generation:

- Carefully read the instructions and notes displayed at the top when you first enter interactive mode.

- If you do not redirect the output of the command to a file, the generated configuration file will simply be printed out.

- You may see messages like "invalid option" and "invalid file" on the bottom left while you are entering the value for a field. This is an artifact of real-time validation. For example, when choosing a training file for ``rsmtool``, the message "invalid file" may be displayed while you navigate to the actual file. Once you get to a valid file, this message should disappear.

- Required fields will *not* accept a blank input (just pressing enter) and will show an error message in the bottom left until a valid input is provided.

- Optional fields will accept blank inputs since they have default values that will be used if no user input is provided. In some cases, default values are shown underlined in parentheses. 

- The configuration files generated interactively contain comments (as indicated by ``// ...``). While RSMTool handles JSON files with comments just fine, you may need to remove the comments manually if you wish to use these files outside of RSMTool.

Non-interactive Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~
For more advanced or experienced users who want to quickly get started with a dummy configuration file
that they feel comfortable editing manually, RSMTool also provides the capability to generate configuration files non-interactively. To do so, simply omit the ``--interactive`` switch in the commands above. For example, to generate a dummy configuration file for ``rsmtool``, the command to run would be:

.. code-block:: bash

    rsmtool generate > dummy_rsmtool.json

When running this command, the following warning would be printed out to stderr:

.. code-block:: bash

    WARNING: Automatically generated configuration files MUST be edited to add values
    for required fields and even for optional ones depending on your data

This warning explains that the generated file *cannot* be used directly as input to ``rsmtool`` since the required fields are filled with dummy values. This can be confirmed by looking at the configuration file the command generates:

.. literalinclude:: assets/dummy_rsmtool.json
    :language: javascript

Note the two comments demarcating the locations of the required and optional fields. Note also that the required fields are filled with the dummy value "ENTER_VALUE_HERE" that *must* be manually edited by the user. The optional fields are filled with default values that may also need to be further edited depending on the data being used.

Just like interactive generation, non-interactive generation is supported by all 5 tools: ``rsmtool``, ``rsmeval``, ``rsmcompare``, ``rsmpredict``, and ``rsmsummarize``. 

Similarly, to include subgroup information in the reports for ``rsmtool``, ``rsmeval``, and ``rsmcompare``, just add ``--subgroups`` to the command. Note that unlike in interactive mode, this would *only* add subgroup-based sections to the ``general_sections`` list in the output file. You will need to manually edit the ``subgroups`` option in the configuration file to enter the subgroup column names.

Generation API
~~~~~~~~~~~~~~
Interactive generation is only meant for end users and can only be used via the 5 command-line tools ``rsmtool``, ``rsmeval``, ``rsmcompare``, ``rsmpredict`` and ``rsmsummarize``. It cannot be used via the RSMTool API.

However, the non-interactive generation can indeed be used via the API which can be useful for more advanced RSMTool users. To illustrate, here's some example Python code to generate a configuration for ``rsmtool`` in the form of a dictionary:

.. code-block:: python

    # import the ConfigurationGenerator class
    from rsmtool.utils.commandline import ConfigurationGenerator

    # instantiate it with the options as needed
    #   we want a dictionary, not a string
    #   we do not want to see any warnings
    #   we want to include subgroup-based sections in the report
    generator = ConfigurationGenerator('rsmtool',
                                       as_string=False,
                                       suppress_warnings=True,
                                       use_subgroups=True)

    # generate the configuration dictionary
    configdict = generator.generate()

    # remember we still need to replace the dummy values
    # for the required fields
    configdict["experiment_id"] = "test_experiment"
    configdict["model"] = "LinearRegression"
    configdict["train_file"] = "train.csv"
    configdict["test_file"] = "test.csv"

    # and don't forget about adding the subgroups
    configdict["subgroups"] = ["GROUP1", "GROUP2"]

    # make other changes to optional fields based on your data
    ...

    # now we can use this dictionary to run an rsmtool experiment via the API
    from rsmtool import run_experiment
    run_experiment(configdict, "/tmp/output")

For more details, refer to the API documentation for the :ref:`ConfigurationGenerator <generation_api>` class.
