import os
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from nose.tools import assert_raises, eq_, ok_

from rsmtool.test_utils import (
    check_file_output,
    check_generated_output,
    check_report,
    collect_warning_messages_from_report,
)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir

# check if tests are being run in strict mode
# if so, any deprecation warnings found in HTML
# reports should not be ignored
STRICT_MODE = os.environ.get('STRICT', None)
IGNORE_DEPRECATION_WARNINGS = False if STRICT_MODE else True


class TestToolCLI:

    @classmethod
    def setUpClass(cls):
        cls.temporary_directories = []
        cls.expected_json_dir = Path(rsmtool_test_dir) / 'data' / 'output'

        common_dir = Path(rsmtool_test_dir) / 'data' / 'experiments'
        cls.rsmtool_config_file = common_dir / 'lr' / 'lr.json'
        cls.rsmeval_config_file = common_dir / 'lr-eval' / 'lr_evaluation.json'
        cls.rsmcompare_config_file = common_dir / 'lr-self-compare' / 'rsmcompare.json'
        cls.rsmpredict_config_file = common_dir / 'lr-predict' / 'rsmpredict.json'
        cls.rsmsummarize_config_file = common_dir / 'lr-self-summary' / 'rsmsummarize.json'
        cls.expected_rsmtool_output_dir = common_dir / 'lr' / 'output'
        cls.expected_rsmeval_output_dir = common_dir / 'lr-eval' / 'output'
        cls.expected_rsmcompare_output_dir = common_dir / 'lr-self-compare' / 'output'
        cls.expected_rsmpredict_output_dir = common_dir / 'lr-predict' / 'output'
        cls.expected_rsmsummarize_output_dir = common_dir / 'lr-self-summary' / 'output'

    @classmethod
    def tearDownClass(cls):
        for tempdir in cls.temporary_directories:
            tempdir.cleanup()

    def check_no_args(self, context):
        """Check running the tool with no arguments."""
        # if the BINPATH environment variable is defined
        # use that to construct the command instead of just
        # the name; this is needed for the CI builds where
        # we do not always activate the conda environment
        binpath = os.environ.get('BINPATH', None)
        if binpath is not None:
            cmd = f"{binpath}/{context}"
        else:
            cmd = f"{context}"

        proc = subprocess.run(shlex.split(cmd, posix='win' not in sys.platform),
                              check=False,
                              stderr=subprocess.DEVNULL,
                              stdout=subprocess.PIPE)
        eq_(proc.returncode, 0)
        ok_(b'usage: ' + bytes(context, encoding="utf-8") in proc.stdout)

    def test_no_args(self):
        # test that the tool without any arguments prints help messag

        # this applies to all tools
        for context in ['rsmtool', 'rsmeval', 'rsmsummarize',
                        'rsmpredict', 'rsmcompare']:
            yield self.check_no_args, context

    def validate_run_output(self, name, experiment_dir):
        """
        Validate output of "run" subcommand for given tool in ``experiment_dir``.

        This method is heavily inspired by the ``rsmtool.test_utils.check_run_*()``
        functions.

        Parameters
        ----------
        name : str
            The name of the tool being tested.
        experiment_dir : str
            Path to rsmtool output directory.
        """
        expected_output_dir = getattr(self, f"expected_{name}_output_dir")

        # all tools except rsmcompare need to have their output files validated
        if name in ['rsmtool', 'rsmeval', 'rsmsummarize', 'rsmpredict']:

            # rsmpredict has its own set of files and it puts them right at the root
            # of the output directory rather than under the "output" subdirectory
            if name == 'rsmpredict':
                output_dir = Path(experiment_dir)
                output_files = [output_dir / 'predictions_with_metadata.csv']
            else:
                output_dir = Path(experiment_dir) / 'output'
                output_files = list(output_dir.glob('*.csv'))

            for output_file in output_files:
                output_filename = output_file.name
                expected_output_file = expected_output_dir / output_filename

                if expected_output_file.exists():
                    check_file_output(str(output_file), str(expected_output_file))

            # we need to do an extra check for rsmtool
            if name == 'rsmtool':
                check_generated_output(list(map(str, output_files)), 'lr', 'rsmtool')

        # there's no report for rsmpredict but for the rest we want
        # the reports to be free of errors and warnings
        if name != 'rsmpredict':
            output_dir = Path(experiment_dir)
            report_dir = output_dir / "report" if name != "rsmcompare" else output_dir
            html_report = list(report_dir.glob('*_report.html'))[0]

            # check report for any errors but ignore warnings
            # which we check below separately
            check_report(html_report, raise_warnings=False)

            # make sure that there are no warnings in the report
            # but ignore deprecation warnings if appropriate
            warning_msgs = collect_warning_messages_from_report(html_report)
            if IGNORE_DEPRECATION_WARNINGS:
                warning_msgs = [msg for msg in warning_msgs if 'DeprecationWarning' not in msg]
            eq_(len(warning_msgs), 0)

    def validate_generate_output(self, name, output, subgroups=False):
        """
        Validate output of "generate" subcommand for given tool in ``experiment_dir``.

        Parameters
        ----------
        name : str
            The name of the tool being tested.
        output : str
            The output of the "generate" subcommand from ``name`` tool
        subgroups : bool, optional
            If ``True``, the ``--subgroups`` was added to the "generate" command
            for ``name``.
            Defaults to ``False``.
        """
        # load the appropriate expected json file and check that its contents
        # match what was printed to stdout with our generate command
        if subgroups:
            expected_json_file = (self.expected_json_dir /
                                  f"autogenerated_{name}_config_groups.json")
        else:
            expected_json_file = (self.expected_json_dir /
                                  f"autogenerated_{name}_config.json")
        with expected_json_file.open('r', encoding='utf-8') as expectedfh:
            expected_output = expectedfh.read().strip()
            eq_(output, expected_output)

    def check_tool_cmd(self, context, subcmd, output_dir=None, working_dir=None):
        """
        Test that the ``subcmd`` invocation for ``context`` works as expected.

        Parameters
        ----------
        context : str
            Name of the tool being tested.
        subcmd : str
            The tool command-line invocation that is being tested.
        output_dir : str, optional
            Directory containing the output for "run" subcommands.
            Will be ``None`` for "generate" subcommands.
            Defaults to ``None``.
        working_dir : str, optional
            If we want the "run" subcommand to be run in a specific
            working directory.
            Defaults to ``None``.
        """
        # if the BINPATH environment variable is defined
        # use that to construct the command instead of just
        # the name; this is needed for the CI builds where
        # we do not always activate the conda environment
        binpath = os.environ.get('BINPATH', None)
        if binpath is not None:
            cmd = f"{binpath}/{context} {subcmd}"
        else:
            cmd = f"{context} {subcmd}"

        # run different checks depending on the given command type
        cmd_type = 'generate' if ' generate' in cmd else 'run'
        if cmd_type == 'run':
            # for run subcommands, we can ignore the messages printed to stdout
            proc = subprocess.run(shlex.split(cmd, posix='win' not in sys.platform),
                                  check=True,
                                  cwd=working_dir,
                                  stderr=subprocess.PIPE,
                                  stdout=subprocess.DEVNULL,
                                  encoding='utf-8')
            # then check that the commmand ran successfully
            ok_(proc.returncode == 0)
            # and, finally, that the output was as expected
            self.validate_run_output(context, output_dir)
        else:
            # for generate subcommands, we ignore the warnings printed to stderr
            subgroups = "--subgroups" in cmd
            proc = subprocess.run(shlex.split(cmd, posix='win' not in sys.platform),
                                  check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  encoding='utf-8')
            ok_(proc.returncode == 0)
            self.validate_generate_output(context,
                                          proc.stdout.strip(),
                                          subgroups=subgroups)

    def test_default_subcommand_is_run(self):
        # test that the default subcommand for all contexts is "run"

        # this applies to all tools
        for context in ['rsmtool', 'rsmeval', 'rsmcompare', 'rsmpredict', 'rsmsummarize']:

            # create a temporary dirextory
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # and test the default subcommand
            config_file = getattr(self, f"{context}_config_file")
            subcmd = f"{config_file} {tempdir.name}"
            yield self.check_tool_cmd, context, subcmd, tempdir.name, None

    def test_run_without_output_directory(self):
        # test that "run" subcommand works without an output directory

        # this applies to all tools except rsmpredict
        for context in ['rsmtool', 'rsmeval', 'rsmcompare', 'rsmsummarize']:

            # create a temporary dirextory
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # and test the run subcommand without an output directory
            config_file = getattr(self, f"{context}_config_file")
            subcmd = f"run {config_file}"
            # we call check_tool_cmd with a working directory here to simulate
            # the usage of the current working directory when the output directory
            # is not specified
            yield self.check_tool_cmd, context, subcmd, tempdir.name, tempdir.name

    def check_run_bad_overwrite(self, cmd):
        """Check that the overwriting error is raised properly."""
        with assert_raises(subprocess.CalledProcessError) as e:
            _ = subprocess.run(shlex.split(cmd, posix='win' not in sys.platform),
                               check=True,
                               stderr=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL)
            ok_('already contains' in e.msg)
            ok_('OSError' in e.msg)

    def test_run_bad_overwrite(self):
        # test that the "run" command fails to overwrite when "-f" is not specified

        # this applies to all tools except rsmpredict and rsmcompare
        for context in ['rsmtool', 'rsmeval', 'rsmsummarize']:

            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # make it look like we ran the tool in this directory already
            os.makedirs(f"{tempdir.name}/output")
            fake_file = Path(tempdir.name) / "output" / "foo.csv"
            fake_file.touch()

            config_file = getattr(self, f"{context}_config_file")
            # if the BINPATH environment variable is defined
            # use that to construct the command instead of just
            # the name; this is needed for the CI builds where
            # we do not always activate the conda environment
            binpath = os.environ.get('BINPATH', None)
            if binpath is not None:
                cmd = f"{binpath}/{context} {config_file} {tempdir.name}"
            else:
                cmd = f"{context} {config_file} {tempdir.name}"
            yield self.check_run_bad_overwrite, cmd

    def test_run_good_overwrite(self):
        #  test that the "run" command does overwrite when "-f" is specified

        # this applies to all tools except rsmpredict and rsmcompare
        for context in ['rsmtool', 'rsmeval', 'rsmsummarize']:

            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # make it look like we ran rsmtool in this directory already
            os.makedirs(f"{tempdir.name}/output")
            fake_file = Path(tempdir.name) / "output" / "foo.csv"
            fake_file.touch()

            config_file = getattr(self, f"{context}_config_file")
            subcmd = f"{config_file} {tempdir.name} -f"
            yield self.check_tool_cmd, context, subcmd, tempdir.name, None

    def test_rsmpredict_run_features_file(self):
        tempdir = TemporaryDirectory()
        self.temporary_directories.append(tempdir)

        subcmd = (f"{self.rsmpredict_config_file} {tempdir.name} "
                  f"--features {tempdir.name}/preprocessed_features.csv")

        self.check_tool_cmd("rsmpredict", subcmd, tempdir.name)

        # check the features file separately
        file1 = Path(tempdir.name) / "preprocessed_features.csv"
        file2 = self.expected_rsmpredict_output_dir / "preprocessed_features.csv"
        check_file_output(str(file1), str(file2))

    def test_generate(self):
        # test that the "generate" subcommand for all tools works as expected
        # in batch mode

        for context in ['rsmtool', 'rsmeval', 'rsmcompare', 'rsmpredict', 'rsmsummarize']:
            yield self.check_tool_cmd, context, "generate", None, None

    def test_generate_with_groups(self):
        # test that the "generate --subgroups" subcommand for all tools works
        # as expected in batch mode

        # this applies to all tools except rsmpredict and rsmsummarize
        for context in ['rsmtool', 'rsmeval', 'rsmcompare']:
            yield self.check_tool_cmd, context, "generate --subgroups", None, None
