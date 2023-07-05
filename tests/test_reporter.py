import unittest
from os.path import join, normpath

from rsmtool.reporter import (
    Reporter,
    comparison_notebook_path,
    master_section_dict,
    notebook_path,
    notebook_path_dict,
    summary_notebook_path,
)

# Since we are in test mode, we want to add a placeholder
# special section to the master section list so that the tests
# will run irrespective of whether or not rsmextra is installed
# We also set a placeholder special_notebook path

for context in ["rsmtool", "rsmeval", "rsmcompare", "rsmsummarize"]:
    master_section_dict["special"][context].append("placeholder_special_section")
    notebook_path_dict["special"].update({context: "special_notebook_path"})

# define the general sections lists to keep tests more readable

general_section_list_rsmtool = master_section_dict["general"]["rsmtool"]
general_section_list_rsmeval = master_section_dict["general"]["rsmeval"]
general_section_list_rsmcompare = master_section_dict["general"]["rsmcompare"]
general_section_list_rsmsummarize = master_section_dict["general"]["rsmsummarize"]


class TestReporter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reporter = Reporter()

    def check_section_lists(self, context):
        general_sections = master_section_dict["general"][context]
        special_sections = master_section_dict["special"][context]
        overlap = set(general_sections) & set(special_sections)
        # check that there are general section
        self.assertTrue(len(general_sections) > 0)
        # check that there is no overlap between general and special section
        # list
        self.assertEqual(len(overlap), 0)

    def test_check_section_lists_rsmtool(self):
        # sanity checks to make sure nothing went wrong when generating
        # master section list
        for context in ["rsmtool", "rsmeval", "rsmcompare"]:
            yield self.check_section_lists, context

    def test_check_section_order_not_enough_sections(self):
        general_sections = ["evaluation", "sysinfo"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["custom.ipynb"]
        subgroups = ["prompt", "gender"]
        section_order = general_sections
        with self.assertRaises(ValueError):
            self.reporter.get_ordered_notebook_files(
                general_sections,
                special_sections=special_sections,
                custom_sections=custom_sections,
                section_order=section_order,
                subgroups=subgroups,
            )

    def test_check_section_order_extra_sections(self):
        general_sections = ["evaluation", "sysinfo"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["custom.ipynb"]
        subgroups = []
        section_order = general_sections + special_sections + custom_sections + ["extra_section"]
        with self.assertRaises(ValueError):
            self.reporter.get_ordered_notebook_files(
                general_sections,
                special_sections=special_sections,
                custom_sections=custom_sections,
                section_order=section_order,
                subgroups=subgroups,
            )

    def test_check_section_order_wrong_sections(self):
        general_sections = ["evaluation", "sysinfo"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["custom.ipynb"]
        subgroups = []
        section_order = ["extra_section1", "extra_section2"]
        with self.assertRaises(ValueError):
            self.reporter.get_ordered_notebook_files(
                general_sections,
                special_sections=special_sections,
                custom_sections=custom_sections,
                section_order=section_order,
                subgroups=subgroups,
            )

    def test_check_section_order(self):
        general_sections = ["evaluation", "sysinfo"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["foobar"]
        section_order = ["foobar"] + special_sections + general_sections
        self.reporter.check_section_order(
            general_sections + special_sections + custom_sections, section_order
        )

    def test_check_general_section_names_rsmtool(self):
        specified_list = ["data_description", "preprocessed_features"]
        self.reporter.check_section_names(specified_list, "general")

    def test_check_general_section_names_wrong_names_1(self):
        specified_list = ["data_description", "feature_stats"]
        with self.assertRaises(ValueError):
            self.reporter.check_section_names(specified_list, "general")

    def test_check_general_section_names_rsmeval_1(self):
        specified_list = ["data_description", "evaluation"]
        self.reporter.check_section_names(specified_list, "general", context="rsmeval")

    def test_check_general_section_names_rsmeval_2(self):
        specified_list = ["data_description", "preprocessed_features"]
        with self.assertRaises(ValueError):
            self.reporter.check_section_names(specified_list, "general", context="rsmeval")

    def test_check_general_section_names_rsmcompare(self):
        specified_list = ["feature_descriptives", "evaluation"]
        self.reporter.check_section_names(specified_list, "general", context="rsmcompare")

    def test_check_general_section_names_wrong_names_2(self):
        specified_list = ["data_description", "evaluation"]
        with self.assertRaises(ValueError):
            self.reporter.check_section_names(specified_list, "general", context="rsmcompare")

    def test_determine_chosen_sections_default_general(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = ["prompt"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(chosen_sections, general_section_list_rsmtool)

    def test_determine_chosen_sections_default_general_no_subgroups(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = []
        no_subgroup_list = [
            s
            for s in general_section_list_rsmtool
            if not s.endswith("by_group") and s != "fairness_analyses"
        ]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(chosen_sections, no_subgroup_list)

    def test_determine_chosen_sections_invalid_general(self):
        general_sections = ["data_description", "foobar"]
        special_sections = []
        custom_sections = []
        subgroups = []
        with self.assertRaises(ValueError):
            chosen_sections = self.reporter.determine_chosen_sections(
                general_sections, special_sections, custom_sections, subgroups
            )
            self.assertEqual(chosen_sections, general_section_list_rsmtool)

    def test_determine_chosen_sections_no_subgroups(self):
        general_sections = ["data_description", "data_description_by_group"]
        special_sections = []
        custom_sections = []
        subgroups = []
        with self.assertRaises(ValueError):
            chosen_sections = self.reporter.determine_chosen_sections(
                general_sections, special_sections, custom_sections, subgroups
            )
            self.assertEqual(chosen_sections, general_section_list_rsmtool)

    def test_determine_chosen_sections_custom_general(self):
        general_sections = ["data_description", "evaluation"]
        special_sections = []
        custom_sections = []
        subgroups = []
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections))

    def test_determine_chosen_sections_default_general_with_special(self):
        general_sections = ["all"]
        special_sections = ["placeholder_special_section"]
        custom_sections = []
        subgroups = ["prompt"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(
            sorted(chosen_sections),
            sorted(general_section_list_rsmtool + special_sections),
        )

    def test_determine_chosen_sections_invalid_special(self):
        general_sections = ["all"]
        special_sections = ["placeholder_special_section", "foobar"]
        custom_sections = []
        subgroups = ["prompt"]
        with self.assertRaises(ValueError):
            self.reporter.determine_chosen_sections(
                general_sections, special_sections, custom_sections, subgroups
            )

    def test_determine_chosen_sections_custom_general_with_special(self):
        general_sections = ["data_description", "evaluation"]
        special_sections = ["placeholder_special_section"]
        custom_sections = []
        subgroups = []
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections + special_sections))

    def test_determine_chosen_sections_default_general_with_subgroups(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = ["prompt", "gender"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_section_list_rsmtool))

    def test_determine_chosen_sections_custom_general_with_special_subgroups_and_custom(
        self,
    ):
        general_sections = ["evaluation", "sysinfo", "evaluation_by_group"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["foobar.ipynb"]
        subgroups = ["prompt", "gender"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections, special_sections, custom_sections, subgroups
        )
        self.assertEqual(
            sorted(chosen_sections),
            sorted(general_sections + special_sections + ["foobar"]),
        )

    def test_determine_chosen_sections_eval_default_general(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = ["prompt"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmeval",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_section_list_rsmeval))

    def test_determine_chosen_sections_eval_custom_general(self):
        general_sections = ["data_description", "consistency"]
        special_sections = []
        custom_sections = []
        subgroups = []
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmeval",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections))

    def test_determine_chosen_sections_eval_default_general_with_no_subgroups(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = []
        no_subgroup_list = [
            s
            for s in general_section_list_rsmeval
            if not s.endswith("by_group") and s != "fairness_analyses"
        ]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmeval",
        )
        self.assertEqual(sorted(chosen_sections), sorted(no_subgroup_list))

    def test_determine_chosen_sections_eval_custom_general_with_special_and_subgroups(
        self,
    ):
        general_sections = [
            "data_description",
            "consistency",
            "data_description_by_group",
        ]
        special_sections = ["placeholder_special_section"]
        custom_sections = []
        subgroups = ["prompt", "gender"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmeval",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections + special_sections))

    def test_determine_chosen_sections_compare_default_general(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = ["prompt"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmcompare",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_section_list_rsmcompare))

    def test_determine_chosen_sections_rsmcompare_custom_general(self):
        general_sections = ["feature_descriptives", "evaluation"]
        special_sections = []
        custom_sections = []
        subgroups = []
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmcompare",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections))

    def test_determine_chosen_sections_rsmcompare_default_general_with_no_subgroups(
        self,
    ):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = []
        no_subgroup_list = [
            s for s in general_section_list_rsmcompare if not s.endswith("by_group")
        ]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmcompare",
        )
        self.assertEqual(sorted(chosen_sections), sorted(no_subgroup_list))

    def test_determine_chosen_sections_rsmcompare_custom_general_with_special_and_subgroups(
        self,
    ):
        general_sections = ["feature_descriptives", "evaluation"]
        special_sections = ["placeholder_special_section"]
        custom_sections = []
        subgroups = ["prompt", "gender"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmcompare",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections + special_sections))

    def test_determine_chosen_sections_rsmsummarize_default_general(self):
        general_sections = ["all"]
        special_sections = []
        custom_sections = []
        subgroups = ["prompt"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmsummarize",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_section_list_rsmsummarize))

    def test_determine_chosen_sections_rsmsummarize_custom_general(self):
        general_sections = ["evaluation"]
        special_sections = []
        custom_sections = []
        subgroups = []
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmsummarize",
        )
        self.assertEqual(sorted(chosen_sections), sorted(general_sections))

    def test_determine_chosen_sections_compare_custom_general_with_special_subgroups_and_custom(
        self,
    ):
        general_sections = ["feature_descriptives", "evaluation"]
        special_sections = ["placeholder_special_section"]
        custom_sections = ["foobar.ipynb"]
        subgroups = ["prompt", "gender"]
        chosen_sections = self.reporter.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context="rsmcompare",
        )
        self.assertEqual(
            sorted(chosen_sections),
            sorted(general_sections + special_sections + ["foobar"]),
        )

    def test_get_ordered_notebook_files_default_rsmtool(self):
        general_sections = ["all"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections, model_type="skll", context="rsmtool"
        )
        no_subgroup_list = [
            s
            for s in general_section_list_rsmtool
            if not s.endswith("by_group") and s != "fairness_analyses"
        ]
        section_list = ["header"] + no_subgroup_list + ["footer"]

        # replace model section with skll_model.
        updated_section_list = [
            "skll_" + sname if sname == "model" else sname for sname in section_list
        ]
        general_section_plus_extension = [s + ".ipynb" for s in updated_section_list]
        expected_notebook_files = [join(notebook_path, s) for s in general_section_plus_extension]
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_custom_rsmtool(self):
        # custom and general sections, custom order and subgroups
        general_sections = ["data_description", "pca", "data_description_by_group"]
        custom_sections = ["/test_path/custom.ipynb"]
        special_sections = ["placeholder_special_section"]
        subgroups = ["prompt"]
        section_order = [
            "custom",
            "data_description",
            "pca",
            "data_description_by_group",
            "placeholder_special_section",
        ]
        special_notebook_path = notebook_path_dict["special"]["rsmtool"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections,
            custom_sections=custom_sections,
            special_sections=special_sections,
            section_order=section_order,
            subgroups=subgroups,
            model_type="skll",
            context="rsmtool",
        )

        expected_notebook_files = (
            [join(notebook_path, "header.ipynb")]
            + ["/test_path/custom.ipynb"]
            + [
                join(notebook_path, s) + ".ipynb"
                for s in ["data_description", "pca", "data_description_by_group"]
            ]
            + [join(special_notebook_path, "placeholder_special_section.ipynb")]
            + [join(notebook_path, "footer.ipynb")]
        )
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_default_rsmeval(self):
        general_sections = ["all"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections, context="rsmeval"
        )
        no_subgroup_list = [
            s
            for s in general_section_list_rsmeval
            if not s.endswith("by_group") and s != "fairness_analyses"
        ]
        section_list = ["header"] + no_subgroup_list + ["footer"]

        general_section_plus_extension = [f"{s}.ipynb" for s in section_list]
        expected_notebook_files = [
            join(notebook_path_dict["general"]["rsmeval"], s)
            for s in general_section_plus_extension
        ]
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_custom_rsmeval(self):
        # custom and general sections, custom order and subgroups

        general_sections = ["evaluation", "consistency", "evaluation_by_group"]
        custom_sections = ["/test_path/custom.ipynb"]
        subgroups = ["prompt"]
        section_order = ["evaluation", "consistency", "custom", "evaluation_by_group"]
        notebook_path = notebook_path_dict["general"]["rsmeval"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections,
            custom_sections=custom_sections,
            section_order=section_order,
            subgroups=subgroups,
            context="rsmeval",
        )

        expected_notebook_files = (
            [join(notebook_path, "header.ipynb")]
            + [join(notebook_path, s) + ".ipynb" for s in ["evaluation", "consistency"]]
            + ["/test_path/custom.ipynb"]
            + [join(notebook_path, "evaluation_by_group.ipynb")]
            + [join(notebook_path, "footer.ipynb")]
        )
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_default_rsmcompare(self):
        general_sections = ["all"]
        comparison_notebook_path = notebook_path_dict["general"]["rsmcompare"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections, context="rsmcompare"
        )
        no_subgroup_list = [
            s for s in general_section_list_rsmcompare if not s.endswith("by_group")
        ]
        section_list = ["header"] + no_subgroup_list + ["footer"]

        general_section_plus_extension = [s + ".ipynb" for s in section_list]
        expected_notebook_files = [
            join(comparison_notebook_path, s) for s in general_section_plus_extension
        ]
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_custom_rsmcompare(self):
        # custom and general sections, custom order and subgroups
        general_sections = [
            "feature_descriptives",
            "score_distributions",
            "features_by_group",
        ]
        custom_sections = ["/test_path/custom.ipynb"]
        subgroups = ["prompt"]
        section_order = [
            "feature_descriptives",
            "score_distributions",
            "custom",
            "features_by_group",
        ]
        comparison_notebook_path = notebook_path_dict["general"]["rsmcompare"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections,
            custom_sections=custom_sections,
            section_order=section_order,
            subgroups=subgroups,
            context="rsmcompare",
        )

        expected_notebook_files = (
            [join(comparison_notebook_path, "header.ipynb")]
            + [
                join(comparison_notebook_path, s) + ".ipynb"
                for s in ["feature_descriptives", "score_distributions"]
            ]
            + ["/test_path/custom.ipynb"]
            + [join(comparison_notebook_path, "features_by_group.ipynb")]
            + [join(comparison_notebook_path, "footer.ipynb")]
        )
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_ordered_notebook_files_custom_rsmsummarize(self):
        # custom and general sections, custom order and subgroups
        general_sections = ["evaluation"]
        custom_sections = ["/test_path/custom.ipynb"]
        subgroups = ["prompt"]
        section_order = ["custom", "evaluation"]
        summary_notebook_path = notebook_path_dict["general"]["rsmsummarize"]
        notebook_files = self.reporter.get_ordered_notebook_files(
            general_sections,
            custom_sections=custom_sections,
            section_order=section_order,
            subgroups=subgroups,
            context="rsmsummarize",
        )

        expected_notebook_files = (
            [join(summary_notebook_path, "header.ipynb")]
            + ["/test_path/custom.ipynb"]
            + [join(summary_notebook_path, s) + ".ipynb" for s in ["evaluation"]]
            + [join(summary_notebook_path, "footer.ipynb")]
        )
        self.assertEqual(notebook_files, expected_notebook_files)

    def test_get_section_file_map_rsmtool(self):
        special_sections = ["placeholder"]
        custom_sections = ["/path/notebook.ipynb"]
        section_file_map = self.reporter.get_section_file_map(
            special_sections, custom_sections, model_type="R"
        )
        self.assertEqual(section_file_map["model"], join(notebook_path, "r_model.ipynb"))
        self.assertEqual(section_file_map["notebook"], "/path/notebook.ipynb")
        self.assertEqual(
            section_file_map["placeholder"],
            normpath("special_notebook_path/placeholder.ipynb"),
        )

    def test_get_section_file_map_rsmeval(self):
        special_sections = ["placeholder"]
        custom_sections = ["/path/notebook.ipynb"]
        section_file_map = self.reporter.get_section_file_map(
            special_sections, custom_sections, context="rsmeval"
        )
        self.assertEqual(
            section_file_map["data_description"],
            join(notebook_path, "data_description.ipynb"),
        )
        self.assertEqual(section_file_map["notebook"], "/path/notebook.ipynb")
        self.assertEqual(
            section_file_map["placeholder"],
            normpath("special_notebook_path/placeholder.ipynb"),
        )

    def test_get_section_file_map_rsmcompare(self):
        special_sections = ["placeholder"]
        custom_sections = ["/path/notebook.ipynb"]
        section_file_map = self.reporter.get_section_file_map(
            special_sections, custom_sections, context="rsmcompare"
        )
        self.assertEqual(
            section_file_map["evaluation"],
            join(comparison_notebook_path, "evaluation.ipynb"),
        )
        self.assertEqual(section_file_map["notebook"], "/path/notebook.ipynb")
        self.assertEqual(
            section_file_map["placeholder"],
            normpath("special_notebook_path/placeholder.ipynb"),
        )

    def test_get_section_file_map_rsmsummarize(self):
        special_sections = ["placeholder"]
        custom_sections = ["/path/notebook.ipynb"]
        section_file_map = self.reporter.get_section_file_map(
            special_sections, custom_sections, context="rsmsummarize"
        )
        self.assertEqual(
            section_file_map["evaluation"],
            join(summary_notebook_path, "evaluation.ipynb"),
        )
        self.assertEqual(section_file_map["notebook"], "/path/notebook.ipynb")
        self.assertEqual(
            section_file_map["placeholder"],
            normpath("special_notebook_path/placeholder.ipynb"),
        )
