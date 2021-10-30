#!/usr/bin/env python
"""
Compare configuration JSON files.

The script takes the reference directory and the test output directory,
identifies all .jsons in the reference directory, finds matching
jsons in the test output directory and compare the content of the two files.

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""

import argparse
import fnmatch
import re
from os import getcwd, linesep, walk
from os.path import abspath, join

from rsmtool.input import check_main_config, parse_json_with_comments

PATH_FIELDS = [
    "train_file",
    "test_file",
    "features",
    "input_features_file",
    "experiment_dir",
    "experiment_dir_old",
    "experiment_dir_new",
    "predictions_file",
    "scale_with",
]


LIST_FIELDS = [
    "feature_prefix",
    "general_sections",
    "special_sections",
    "custom_sections",
    "subgroups",
    "section_order",
]


def find_jsons(json_dir):
    """Find all JSON files in given directory."""
    dir_content = walk(json_dir)
    json_file_dict = {}
    for dirpath, ___, filenames in dir_content:
        for filename in fnmatch.filter(filenames, "*.json"):

            # we are not interested in feature json files
            if dirpath == "Feature":
                continue
            json_full_path = join(dirpath, filename)

            # check that we don't already have a file with this name
            if filename in json_file_dict:
                print(
                    f"Duplicate json file name: {json_file_dict[filename]} and {json_full_path}"
                )

            json_file_dict[filename] = json_full_path
    return json_file_dict


def get_dict_differences(ref_json, test_json, ref_file, test_file):
    """
    Identify differences between the two JSON dictionaries.

    Parameters
    ----------
    ref_json : dict
        Refernce JSON configuration dictionary.
    test_json : dict
        Test JSON configuration dictionary.
    ref_file : str
        Name of reference configuration file.
    test_file : str
        Name of test configuration file.

    Returns
    -------
    result : list of str
        List containing comparison results.
    """
    result = [f"REF: {ref_file}, TEST: {test_file}"]
    for key in ref_json:
        if key not in test_json:
            result.append(f"Field {key} not specified in test json")
        elif ref_json[key] and test_json[key] is not None:
            result.append(
                f"Field {key} is set to None in test json and to {ref_json[key]} in reference file"
            )
        elif ref_json[key] is not None and test_json[key]:
            result.append(
                f"Field {key} is set to None in reference json and to {test_json[key]} in test file"
            )
        elif not ref_json[key] == test_json[key]:
            result.append(
                f"Field {key} has different value in test json {ref_json[key]} vs {test_json[key]}"
            )
    added_fields = set(test_json.keys()).difference(set(ref_json.keys()))
    if len(added_fields) > 0:
        result.append(
            f"The following extra fields are present in test json: {', '.join(added_fields)}"
        )

    return result


def compare_jsons(ref_dir, test_dir):
    """Compare the configuration JSONs in the two given directories."""
    ref_json_dict = find_jsons(ref_dir)
    test_json_dict = find_jsons(test_dir)
    result_dict = {}
    for ref_json in ref_json_dict:
        tool = ref_json.split("_")[-1].rstrip(".json")
        # the tool appends the name of the tool to the output file
        # so we need to account for this since the name will be duplicated

        test_json = f"{ref_json.rstrip('.json')}_{tool}.json"

        print(f"{ref_json} vs. {test_json}")
        if test_json not in test_json_dict:
            result_dict[ref_json] = "missing"
        else:
            # read the files
            ref_json_obj = parse_json_with_comments(ref_json_dict[ref_json])
            test_json_obj = parse_json_with_comments(test_json_dict[test_json])

            # add in the default values
            ref_json_norm = check_main_config(ref_json_obj, context=tool)
            test_json_norm = check_main_config(test_json_obj, context=tool)

            # convert paths in reference files to absolute paths and replace
            # My Documents with Documents
            for field in PATH_FIELDS:
                if (
                    field in ref_json_norm
                    and ref_json_norm[field] is not None
                    and not ref_json_norm[field] == "asis"
                ):
                    new_ref_path = abspath(join(ref_dir, ref_json_norm[field]))
                    new_ref_path = re.sub("My Documents", "Documents", new_ref_path)
                    ref_json_norm[field] = new_ref_path
            # sort the values in list fields
            for field in LIST_FIELDS:
                if field in ref_json_norm and ref_json_norm[field] is not None:
                    ref_json_norm[field] = sorted(ref_json_norm[field])
                if field in test_json_norm and test_json_norm[field] is not None:
                    test_json_norm[field] = sorted(test_json_norm[field])
            # compare two jsons
            if not ref_json_norm == test_json_norm:
                result_dict[ref_json] = get_dict_differences(
                    ref_json_norm,
                    test_json_norm,
                    ref_json_dict[ref_json],
                    test_json_dict[test_json],
                )
            else:
                result_dict[ref_json] = "OK"
    return result_dict


def print_result(result_dict):
    """Print overall comparison results."""
    matched = sorted([json for (json, result) in result_dict.items() if result == "OK"])
    missing = sorted(
        [json for (json, result) in result_dict.items() if result == "missing"]
    )
    discrepant = [
        json for json in result_dict if json not in matched and json not in missing
    ]
    print(f"Total reference files: {len(result_dict)}")
    print("------------------------")
    print(f"Matched: {linesep} {linesep.join(matched)}")
    print("------------------------")
    print(f"Missing test files: {linesep} {linesep.join(missing)}")
    print("------------------------")
    print("Discrepant")
    for json in discrepant:
        print(linesep, json, linesep, linesep.join(result_dict[json]))
    print(f"Total reference files: {len(result_dict)}")
    print(f"Total matched: {len(matched)}")
    print(f"Total missing: {len(missing)}")
    print(f"Total discrepant: {len(discrepant)}")


def main():  # noqa: D103
    # set up an argument parser
    parser = argparse.ArgumentParser(prog="compare_config_jsons.py")
    parser.add_argument(dest="ref_dir", help="Parent directory containing .json files")
    parser.add_argument(
        dest="test_dir",
        help="Parent directory containing " "the outputs of RSMApp",
        default=getcwd(),
        nargs="?",
    )

    # parse given command line arguments
    args = parser.parse_args()
    result_dict = compare_jsons(args.ref_dir, args.test_dir)
    print_result(result_dict)


if __name__ == "__main__":
    main()
