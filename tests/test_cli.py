import logging
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

from rsmtool.rsmcompare import main as rsmcompare_main
from rsmtool.rsmeval import main as rsmeval_main
from rsmtool.rsmexplain import main as rsmexplain_main
from rsmtool.rsmpredict import main as rsmpredict_main
from rsmtool.rsmsummarize import main as rsmsummarize_main
from rsmtool.rsmtool import main as rsmtool_main
from rsmtool.rsmxval import main as rsmxval_main
from rsmtool.test_utils import (
    check_file_output,
    check_generated_output,
    check_report,
    collect_warning_messages_from_report,
)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir

# check if tests are being run in strict mode
# if so, any warnings found in HTML
# reports should not be ignored
STRICT_MODE = os.environ.get("STRICT", None)
IGNORE_WARNINGS = False if STRICT_MODE else True

# create a dictionary mapping tool names to their main functions
# NOTE: technically we could have just used `globals()` instead
# of this dictionary, but that would not be as safe
MAIN_FUNCTIONS = {
    "rsmtool": rsmtool_main,
    "rsmeval": rsmeval_main,
    "rsmcompare": rsmcompare_main,
    "rsmpredict": rsmpredict_main,
    "rsmsummarize": rsmsummarize_main,
    "rsmxval": rsmxval_main,
    "rsmexplain": rsmexplain_main,
}


class TestToolCLI(unittest.TestCase):
    """Test class for ToolCLI tests."""

    temporary_directories = []
    temporary_files = []
    expected_json_dir = Path(rsmtool_test_dir) / "data" / "output"

    common_dir = Path(rsmtool_test_dir) / "data" / "experiments"
    rsmtool_config_file = common_dir / "lr" / "lr.json"
    rsmeval_config_file = common_dir / "lr-eval" / "lr_evaluation.json"
    rsmcompare_config_file = common_dir / "lr-self-compare" / "rsmcompare.json"
    rsmpredict_config_file = common_dir / "lr-predict" / "rsmpredict.json"
    rsmsummarize_config_file = common_dir / "lr-self-summary" / "rsmsummarize.json"
    rsmxval_config_file = common_dir / "lr-xval" / "lr_xval.json"
    rsmexplain_config_file = common_dir / "svr-explain" / "rsmexplain.json"
    expected_rsmtool_output_dir = common_dir / "lr" / "output"
    expected_rsmeval_output_dir = common_dir / "lr-eval" / "output"
    expected_rsmcompare_output_dir = common_dir / "lr-self-compare" / "output"
    expected_rsmpredict_output_dir = common_dir / "lr-predict" / "output"
    expected_rsmsummarize_output_dir = common_dir / "lr-self-summary" / "output"
    expected_rsmxval_output_dir = common_dir / "lr-xval" / "output"
    expected_rsmexplain_output_dir = common_dir / "svr-explain" / "output"

    @classmethod
    def tearDownClass(cls):
        for tempdir in cls.temporary_directories:
            rmtree(tempdir.name, ignore_errors=True)
        for tempfile in cls.temporary_files:
            os.unlink(tempfile)

    def check_no_args(self, context):
        """Check running the tool with no arguments."""
        # call the main function for the given context
        # with no arguments and check that it exits
        # with exit code 0 and prints out the usage
        main_function = MAIN_FUNCTIONS[context]
        with redirect_stdout(StringIO()) as captured_stdout:
            with self.assertRaises(SystemExit) as excm:
                main_function([])

        # check that the exit code is 0
        self.assertEqual(excm.exception.code, 0)

        # check that the appropriate usage was printed out
        self.assertTrue(f"usage: {context}" in captured_stdout.getvalue().lower())

    def test_no_args(self):
        # test that the tool without any arguments prints help messag

        # this applies to all tools
        for context in [
            "rsmtool",
            "rsmeval",
            "rsmsummarize",
            "rsmpredict",
            "rsmcompare",
            "rsmxval",
            "rsmexplain",
        ]:
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
        if name in ["rsmtool", "rsmeval", "rsmsummarize", "rsmpredict", "rsmxval", "rsmexplain"]:
            # rsmpredict has its own set of files and it puts them right at the root
            # of the output directory rather than under the "output" subdirectory;
            # rsmxval also needs to be specially handled
            if name == "rsmxval":
                self.validate_run_output_rsmxval(experiment_dir)
            elif name == "rsmpredict":
                output_dir = Path(experiment_dir)
                output_files = [output_dir / "predictions_with_metadata.csv"]
            else:
                output_dir = Path(experiment_dir) / "output"
                output_files = list(output_dir.glob("*.csv"))

                for output_file in output_files:
                    output_filename = output_file.name
                    expected_output_file = expected_output_dir / output_filename

                    if expected_output_file.exists():
                        check_file_output(str(output_file), str(expected_output_file))

                # we need to do an extra check for rsmtool
                if name == "rsmtool":
                    check_generated_output(list(map(str, output_files)), "lr", "rsmtool")

        # there's no report for rsmpredict but for the rest we want
        # the reports to be free of errors and warnings; for rsmxval
        # there are multiple reports to check
        if name in ["rsmtool", "rsmeval", "rsmcompare", "rsmsummarize", "rsmxval", "rsmexplain"]:
            output_dir = Path(experiment_dir)
            if name == "rsmxval":
                folds_dir = output_dir / "folds"
                per_fold_html_reports = list(
                    map(str, folds_dir.glob("??/report/lr_xval_fold??.html"))
                )
                evaluation_report = (
                    output_dir / "evaluation" / "report" / "lr_xval_evaluation_report.html"
                )
                summary_report = (
                    output_dir / "fold-summary" / "report" / "lr_xval_fold_summary_report.html"
                )
                final_model_report = (
                    output_dir / "final-model" / "report" / "lr_xval_model_report.html"
                )
                html_reports = per_fold_html_reports + [
                    evaluation_report,
                    summary_report,
                    final_model_report,
                ]
            else:
                report_dir = output_dir / "report" if name != "rsmcompare" else output_dir
                html_reports = report_dir.glob("*_report.html")

            # check reports for any errors but ignore warnings
            # which we check below separately
            for html_report in html_reports:
                check_report(html_report, raise_warnings=False)

                # make sure that there are no warnings in the report
                # but ignore warnings if in STRICT mode
                if not IGNORE_WARNINGS:
                    warning_msgs = collect_warning_messages_from_report(html_report)
                    self.assertEqual(len(warning_msgs), 0)

    def validate_run_output_rsmxval(self, experiment_dir):
        output_dir = Path(experiment_dir)
        expected_output_dir = getattr(self, "expected_rsmxval_output_dir")

        # first check that each fold's rsmtool output is as expected
        actual_folds_dir = output_dir / "folds"
        expected_folds_dir = expected_output_dir / "folds"
        fold_output_files = list(actual_folds_dir.glob("??/output/*.csv"))
        fold_nums = set()
        for fold_output_file in fold_output_files:
            fold_output_filename = fold_output_file.relative_to(actual_folds_dir)
            fold_nums.add(fold_output_filename.parts[0])
            expected_fold_output_file = expected_folds_dir / fold_output_filename

            if expected_fold_output_file.exists():
                check_file_output(str(fold_output_file), str(expected_fold_output_file))

        for fold_num in fold_nums:
            check_generated_output(
                list(map(str, fold_output_files)), f"lr_xval_fold{fold_num}", "rsmtool"
            )

        # next check that the evaluation output is as expected
        actual_eval_output_dir = output_dir / "evaluation"
        expected_eval_output_dir = expected_output_dir / "evaluation"

        eval_output_files = actual_eval_output_dir.glob("output/*.csv")
        for eval_output_file in eval_output_files:
            eval_output_filename = eval_output_file.relative_to(actual_eval_output_dir)
            expected_eval_output_file = expected_eval_output_dir / eval_output_filename

            if expected_eval_output_file.exists():
                check_file_output(str(eval_output_file), str(expected_eval_output_file))

        # next check that the summary output is as expected
        actual_summary_output_dir = output_dir / "fold-summary"
        expected_summary_output_dir = expected_output_dir / "fold-summary"

        summary_output_files = actual_summary_output_dir.glob("output/*.csv")
        for summary_output_file in summary_output_files:
            summary_output_filename = summary_output_file.relative_to(actual_summary_output_dir)
            expected_summary_output_file = expected_summary_output_dir / summary_output_filename

            if expected_summary_output_file.exists():
                check_file_output(str(summary_output_file), str(expected_summary_output_file))

        # next check that the final model rsmtool output is as expected
        actual_final_model_output_dir = output_dir / "final-model"
        expected_final_model_output_dir = expected_output_dir / "final-model"

        final_model_output_files = list(actual_final_model_output_dir.glob("output/*.csv"))
        for final_model_output_file in final_model_output_files:
            final_model_output_filename = final_model_output_file.relative_to(
                actual_final_model_output_dir
            )
            expected_final_model_output_file = (
                expected_final_model_output_dir / final_model_output_filename
            )

            if expected_final_model_output_file.exists():
                check_file_output(
                    str(final_model_output_file), str(expected_final_model_output_file)
                )

        check_generated_output(final_model_output_files, "lr_xval_model", "rsmtool")

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
            expected_json_file = self.expected_json_dir / f"autogenerated_{name}_config_groups.json"
        else:
            expected_json_file = self.expected_json_dir / f"autogenerated_{name}_config.json"
        with expected_json_file.open("r", encoding="utf-8") as expectedfh:
            expected_output = expectedfh.read().strip()
            self.assertEqual(output, expected_output)

    def check_tool_cmd(self, context, args, output_dir=None, generate_output_file=None):
        """
        Test that the invocation for ``context`` with ``args`` works as expected.

        Parameters
        ----------
        context : str
            Name of the tool being tested.
        args : List[str]
            The list of arguments to be passed to the tool.
        output_dir : str, optional
            Directory containing the output for "run" subcommands.
            Will be ``None`` for "generate" subcommands.
            Defaults to ``None``.
        working_dir : str, optional
            If we want the "run" subcommand to be run in a specific
            working directory.
            Defaults to ``None``.
        generate_output_file: str, optional
           If we want the output of the "generate" subcommand to be
           written to a specific output file.
           Defaults to ``None``.
        """
        # get the main function for the given context
        main_function = MAIN_FUNCTIONS[context]

        # increase the logging level for the tool's logger since we don't
        # want to see the logging messages when we run the tool for this test
        context_logger = logging.getLogger(f"rsmtool.{context}")
        context_logger.setLevel(logging.CRITICAL)

        # do the same for the `utils.commandline` logger since we don't want to
        # see the logging messages for configuration generation either
        commandline_logger = logging.getLogger("rsmtool.utils.commandline")
        commandline_logger.setLevel(logging.CRITICAL)

        # call the main function for the given context with the arguments
        # but redirect STDOUT and STDERR to StringIO objects
        with redirect_stdout(StringIO()) as captured_stdout, redirect_stderr(StringIO()):
            main_function(args)

        # restore the logging level for both loggers
        context_logger.setLevel(logging.INFO)
        commandline_logger.setLevel(logging.INFO)

        # run different checks depending on the given command type
        cmd_type = "generate" if "generate" in args else "run"
        if cmd_type == "run":
            # check that the output generated is as expected
            self.validate_run_output(context, output_dir)
        else:
            subgroups = "--subgroups" in args
            # if an output file was specified, we get the output to validate
            # by reading that in instead of from STDOUT; in that case STDOUT
            # only contains a message which we can also validate
            if generate_output_file:
                with open(generate_output_file, "r") as outfh:
                    output_to_check = outfh.read().strip()
            else:
                output_to_check = captured_stdout.getvalue().strip()

            self.validate_generate_output(context, output_to_check, subgroups=subgroups)

    def test_default_subcommand_is_run(self):
        # test that the default subcommand for all contexts is "run"

        # this applies to all tools
        for context in [
            "rsmtool",
            "rsmeval",
            "rsmcompare",
            "rsmpredict",
            "rsmsummarize",
            "rsmxval",
            "rsmexplain",
        ]:
            # create a temporary dirextory
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # and test the default subcommand
            config_file = getattr(self, f"{context}_config_file")
            args = [str(config_file), tempdir.name]
            yield self.check_tool_cmd, context, args, tempdir.name, None

    def check_run_without_output_directory(self, context, args):
        """
        Check that the "run" subcommand works without an output directory.

        Parameters
        ----------
        context : str
            Name of the tool being tested.
        args : List[str]
            The list of arguments to be passed to the tool.
        """
        # get the current working directory
        current_dir = os.getcwd()

        # patch the appropriate runner function for the tool being tested
        if context == "rsmtool":
            patcher = patch("rsmtool.rsmtool.run_experiment")
        elif context == "rsmeval":
            patcher = patch("rsmtool.rsmeval.run_evaluation")
        elif context == "rsmcompare":
            patcher = patch("rsmtool.rsmcompare.run_comparison")
        elif context == "rsmsummarize":
            patcher = patch("rsmtool.rsmsummarize.run_summary")
        elif context == "rsmxval":
            patcher = patch("rsmtool.rsmxval.run_cross_validation")
        elif context == "rsmexplain":
            patcher = patch("rsmtool.rsmexplain.generate_explanation")

        # start the patcher
        mocked_runner = patcher.start()

        # increase the logging level for the tool's logger since we don't
        # want to see the logging messages when we run the tool for this test
        context_logger = logging.getLogger(f"rsmtool.{context}")
        context_logger.setLevel(logging.CRITICAL)

        # call the main function for the given context with the arguments
        main_function = MAIN_FUNCTIONS[context]
        main_function(args)

        # restore the logging level for the tool's logger
        context_logger.setLevel(logging.INFO)

        # stop the patcher
        patcher.stop()

        # check that the runner was called with the right arguments
        # including the current working directory as the output directory
        config_file = mocked_runner.call_args[0][0]
        output_dir = mocked_runner.call_args[0][1]
        self.assertEqual(config_file, str(config_file))
        self.assertEqual(output_dir, current_dir)

    def test_run_without_output_directory(self):
        # test that "run" subcommand works without an output directory

        # this applies to all tools except rsmpredict
        for context in [
            "rsmtool",
            "rsmeval",
            "rsmcompare",
            "rsmsummarize",
            "rsmxval",
            "rsmexplain",
        ]:
            # create a temporary dirextory
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # and test the run subcommand without an output directory
            config_file = getattr(self, f"{context}_config_file")
            args = ["run", str(config_file)]
            yield self.check_run_without_output_directory, context, args

    def check_run_bad_overwrite(self, context, args):
        """Check that the overwriting error is raised properly."""
        # increase the logging level for the tool's logger since we don't
        # want to see the logging messages when we run the tool for this test
        context_logger = logging.getLogger(f"rsmtool.{context}")
        context_logger.setLevel(logging.CRITICAL)

        # run the command and check that it fails with the expected error that
        # the output directory already contains a non-empty output sub-directory
        with self.assertRaisesRegex(OSError, "already contains a non-empty"):
            main_function = MAIN_FUNCTIONS[context]
            main_function(args)

        # restore the logging level for the tool's logger
        context_logger.setLevel(logging.INFO)

    def test_run_bad_overwrite(self):
        # test that the "run" command fails to overwrite when "-f" is not specified
        # this applies to all tools except rsmpredict, rsmcompare, and rsmxval
        for context in ["rsmtool", "rsmeval", "rsmsummarize", "rsmexplain"]:
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # make it look like we ran the tool in this directory already
            os.makedirs(f"{tempdir.name}/output")
            fake_file = Path(tempdir.name) / "output" / "foo.csv"
            fake_file.touch()

            # now run the tool again and check that it fails
            config_file = getattr(self, f"{context}_config_file")
            args = [str(config_file), tempdir.name]
            yield self.check_run_bad_overwrite, context, args

    def test_run_good_overwrite(self):
        #  test that the "run" command does overwrite when "-f" is specified

        # this applies to all tools except rsmpredict, rsmcompare, and rsmxval
        for context in ["rsmtool", "rsmeval", "rsmsummarize", "rsmexplain"]:
            tempdir = TemporaryDirectory()
            self.temporary_directories.append(tempdir)

            # make it look like we ran rsmtool in this directory already
            os.makedirs(f"{tempdir.name}/output")
            fake_file = Path(tempdir.name) / "output" / "foo.csv"
            fake_file.touch()

            config_file = getattr(self, f"{context}_config_file")
            args = [str(config_file), tempdir.name, "-f"]
            yield self.check_tool_cmd, context, args, tempdir.name, None

    def test_rsmpredict_run_features_file(self):
        tempdir = TemporaryDirectory()
        self.temporary_directories.append(tempdir)

        args = [
            str(self.rsmpredict_config_file),
            tempdir.name,
            "--features",
            f"{tempdir.name}/preprocessed_features.csv",
        ]

        self.check_tool_cmd("rsmpredict", args, tempdir.name, None)

        # check the features file separately
        file1 = Path(tempdir.name) / "preprocessed_features.csv"
        file2 = self.expected_rsmpredict_output_dir / "preprocessed_features.csv"
        check_file_output(str(file1), str(file2))

    def test_generate(self):
        # test that the "generate" subcommand for all tools works as expected
        # in batch mode

        for context in [
            "rsmtool",
            "rsmeval",
            "rsmcompare",
            "rsmpredict",
            "rsmsummarize",
            "rsmxval",
            "rsmexplain",
        ]:
            yield self.check_tool_cmd, context, ["generate"], None, None

    def test_generate_with_groups(self):
        # test that the "generate --subgroups" subcommand for all tools works
        # as expected in batch mode

        # this applies to all tools except rsmpredict, rsmsummarize,
        # rsmxval and rsmexplain; rsmxval does support subgroups but since it has no sections
        # fields, it's not relevant for this test
        for context in ["rsmtool", "rsmeval", "rsmcompare"]:
            yield self.check_tool_cmd, context, ["generate", "--subgroups"], None, None

    def test_generate_with_output_file(self):
        # test that the "generate --output <file>" subcommand for all tools works
        # as expected in batch mode
        for context in [
            "rsmtool",
            "rsmeval",
            "rsmcompare",
            "rsmpredict",
            "rsmsummarize",
            "rsmxval",
            "rsmexplain",
        ]:
            tempf = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tempf.close()
            self.temporary_files.append(tempf.name)
            yield (
                self.check_tool_cmd,
                context,
                ["generate", "--output", tempf.name],
                None,
                tempf.name,
            )

    def test_generate_with_output_file_and_groups(self):
        # test that the "generate --subgroups" subcommand for all tools works
        # as expected when written to output files

        # this applies to all tools except rsmpredict, rsmsummarize,
        # rsmxval and rsmexplain; rsmxval does support subgroups but
        # since it has no sections fields, it's not relevant for this test
        for context in ["rsmtool", "rsmeval", "rsmcompare"]:
            tempf = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tempf.close()
            self.temporary_files.append(tempf.name)
            yield (
                self.check_tool_cmd,
                context,
                ["generate", "--subgroups", "--output", tempf.name],
                None,
                tempf.name,
            )
