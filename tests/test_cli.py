import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

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
            tempdir.cleanup()
        for tempfile in cls.temporary_files:
            os.unlink(tempfile)

    def check_no_args(self, context):
        """Check running the tool with no arguments."""
        # if the BINPATH environment variable is defined
        # use that to construct the command instead of just
        # the name; this is needed for the CI builds where
        # we do not always activate the conda environment
        binpath = os.environ.get("BINPATH", None)
        if binpath is not None:
            cmd = f"{binpath}/{context}"
        else:
            cmd = f"{context}"

        proc = subprocess.run(
            shlex.split(cmd, posix="win" not in sys.platform),
            check=False,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(b"usage: " + bytes(context, encoding="utf-8") in proc.stdout)

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

    def check_tool_cmd(
        self, context, subcmd, output_dir=None, working_dir=None, generate_output_file=None
    ):
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
        generate_output_file: str, optional
           If we want the output of the "generate" subcommand to be
           written to a specific output file.
           Defaults to ``None``.
        """
        # if the BINPATH environment variable is defined
        # use that to construct the command instead of just
        # the name; this is needed for the CI builds where
        # we do not always activate the conda environment
        binpath = os.environ.get("BINPATH", None)
        if binpath is not None:
            cmd = f"{binpath}/{context} {subcmd}"
        else:
            cmd = f"{context} {subcmd}"

        # run different checks depending on the given command type
        cmd_type = "generate" if " generate" in cmd else "run"
        if cmd_type == "run":
            # for run subcommands, we can ignore the messages printed to stdout
            proc = subprocess.run(
                shlex.split(cmd, posix="win" not in sys.platform),
                check=True,
                cwd=working_dir,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                encoding="utf-8",
            )
            # then check that the commmand ran successfully
            self.assertTrue(proc.returncode == 0)
            # and, finally, that the output was as expected
            self.validate_run_output(context, output_dir)
        else:
            # for generate subcommands, we ignore the warnings printed to stderr
            subgroups = "--subgroups" in cmd
            proc = subprocess.run(
                shlex.split(cmd, posix="win" not in sys.platform),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            self.assertTrue(proc.returncode == 0)
            # if an output file was specified, we get the output to validate
            # by reading that in instead of from STDOUT; in that case STDOUT
            # only contains a message which we can also validate
            if generate_output_file:
                with open(generate_output_file, "r") as outfh:
                    output_to_check = outfh.read().strip()
            else:
                output_to_check = proc.stdout.strip()

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
            subcmd = f"{config_file} {tempdir.name}"
            yield self.check_tool_cmd, context, subcmd, tempdir.name, None, None

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
            subcmd = f"run {config_file}"
            # we call check_tool_cmd with a working directory here to simulate
            # the usage of the current working directory when the output directory
            # is not specified
            yield self.check_tool_cmd, context, subcmd, tempdir.name, tempdir.name, None

    def check_run_bad_overwrite(self, cmd):
        """Check that the overwriting error is raised properly."""
        with self.assertRaises(subprocess.CalledProcessError) as e:
            _ = subprocess.run(
                shlex.split(cmd, posix="win" not in sys.platform),
                check=True,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            self.assertTrue("already contains" in e.msg)
            self.assertTrue("OSError" in e.msg)

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

            config_file = getattr(self, f"{context}_config_file")
            # if the BINPATH environment variable is defined
            # use that to construct the command instead of just
            # the name; this is needed for the CI builds where
            # we do not always activate the conda environment
            binpath = os.environ.get("BINPATH", None)
            if binpath is not None:
                cmd = f"{binpath}/{context} {config_file} {tempdir.name}"
            else:
                cmd = f"{context} {config_file} {tempdir.name}"
            yield self.check_run_bad_overwrite, cmd

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
            subcmd = f"{config_file} {tempdir.name} -f"
            yield self.check_tool_cmd, context, subcmd, tempdir.name, None, None

    def test_rsmpredict_run_features_file(self):
        tempdir = TemporaryDirectory()
        self.temporary_directories.append(tempdir)

        subcmd = (
            f"{self.rsmpredict_config_file} {tempdir.name} "
            f"--features {tempdir.name}/preprocessed_features.csv"
        )

        self.check_tool_cmd("rsmpredict", subcmd, tempdir.name, None)

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
            yield self.check_tool_cmd, context, "generate", None, None, None

    def test_generate_with_groups(self):
        # test that the "generate --subgroups" subcommand for all tools works
        # as expected in batch mode

        # this applies to all tools except rsmpredict, rsmsummarize,
        # rsmxval and rsmexplain; rsmxval does support subgroups but since it has no sections
        # fields, it's not relevant for this test
        for context in ["rsmtool", "rsmeval", "rsmcompare"]:
            yield self.check_tool_cmd, context, "generate --subgroups", None, None, None

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
                f"generate --output {tempf.name}",
                None,
                None,
                tempf.name,
            )

    def test_generate_with_output_file_and_groups(self):
        # test that the "generate --subgroups" subcommand for all tools works
        # as expected when written to output files

        # this applies to all tools except rsmpredict, rsmsummarize,
        # rsmxval and rsmexplain; rsmxval does support subgroups but since it has no sections
        # fields, it's not relevant for this test
        for context in ["rsmtool", "rsmeval", "rsmcompare"]:
            tempf = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tempf.close()
            self.temporary_files.append(tempf.name)
            yield (
                self.check_tool_cmd,
                context,
                f"generate --subgroups --output {tempf.name}",
                None,
                None,
                tempf.name,
            )
