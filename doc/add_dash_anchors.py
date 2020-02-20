#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add Dash-style anchors to already-generated HTML documentation.

This script iterates over pre-specified HTML files generated via
sphinx-build, finds all of the sections, and adds Dash-style anchors
so that when those HTML files are displayed in the Dash macOS app,
the sections are displayed in a Dash TOC on the right.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import argparse
import logging
import re
import pathlib
import unicodedata
from urllib.parse import quote

from bs4 import BeautifulSoup

# pre-define the list of HTML files we want to miodify
FILES_TO_MODIFY = ["advanced_usage.html", "api.html", "contributing.html",
                   "custom_notebooks.html", "evaluation.html",
                   "getting_started.html", "pipeline.html",
                   "internal.html", "tutorial.html",
                   "usage_rsmtool.html", "utilities.html", "who.html"]
PILCROW = unicodedata.lookup('PILCROW SIGN')


def main():  # noqa: D103

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='add_dash_anchors.py')
    parser.add_argument("htmldir",
                        type=pathlib.Path,
                        help="path to the already-built HTML documentation")

    # parse given command line arguments
    args = parser.parse_args()

    # set up the logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # iterate over all the built HTML files
    for htmlfile in args.htmldir.glob("**/*.html"):

        # we only care about the pre-specified html files
        if htmlfile.name in FILES_TO_MODIFY:

            logging.info(f'Processing {htmlfile.name} ...')

            # parse the file
            with open(htmlfile, 'r') as htmlfh:
                soup = BeautifulSoup(htmlfh, features='html.parser')

            # each HTML file has a main section which we do not need
            # but we need _all_ of the other sections
            sections = soup.body.div.find_all("div", class_="section")[1:]
            for section in sections:
                section_title = section.find(re.compile(r'^h[0-9]')).text
                section_title = section_title.rstrip(PILCROW)

                # convert this title to percent-encoded format which will be
                # the name of our entry
                entry_name = quote(section_title)
                entry_type = 'Section'
                anchor_name = f"//apple_ref/cpp/{entry_type}/{entry_name}"

                # create a new anchor tag for this subsection
                anchor_tag = soup.new_tag('a',
                                          attrs={'name': anchor_name,
                                                 'class': "dashAnchor"})

                # insert this new tag right before the section
                section.insert_before(anchor_tag)

            # overwrite the original HTML file
            with open(htmlfile, 'w') as outfh:
                outfh.write(str(soup))


if __name__ == '__main__':
    main()
