#!/usr/bin/env python
from setuptools import find_packages, setup

# Get version without importing, which avoids dependency issues
exec(compile(open("rsmtool/version.py").read(), "rsmtool/version.py", "exec"))
# (we use the above instead of execfile for Python 3.x compatibility)


def readme():
    with open("README.rst") as f:
        return f.read()


def requirements():
    req_path = "requirements.txt"
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(
    name="rsmtool",
    version=__version__,  # noqa
    description="Rater scoring modeling tool",
    long_description=readme(),
    keywords="scoring modeling",
    url="http://github.com/EducationalTestingService/rsmtool",
    maintainer="Nitin Madnani",
    maintainer_email="nmadnani@ets.org",
    license="Apache 2",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "rsmtool = rsmtool.rsmtool:main",
            "rsmeval = rsmtool.rsmeval:main",
            "rsmpredict = rsmtool.rsmpredict:main",
            "rsmcompare = rsmtool.rsmcompare:main",
            "rsmsummarize = rsmtool.rsmsummarize:main",
            "rsmxval = rsmtool.rsmxval:main",
            "rsmexplain = rsmtool.rsmexplain:main",
            "render_notebook = rsmtool.reporter:main",
            "convert_feature_json = rsmtool.convert_feature_json:main",
        ]
    },
    install_requires=requirements(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
