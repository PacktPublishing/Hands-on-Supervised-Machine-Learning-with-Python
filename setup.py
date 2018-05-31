# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
import setuptools

with open("packtml/VERSION", 'r') as vsn:
    VERSION = vsn.read().strip()

# Permitted args: "install" only, basically.
UNSUPPORTED_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed', 'test', 'build_ext'
}

intersect = UNSUPPORTED_COMMANDS.intersection(set(sys.argv))
if intersect:
    msg = "The following arguments are unsupported: %s. " \
          "To install, please use `python setup.py install`." \
          % str(list(intersect))

    # if "test" is in the arguments, make sure the user knows how to test.
    if "test" in intersect:
        msg += " To test, make sure pytest is installed, and after " \
               "installation run `pytest packtml`"

    raise ValueError(msg)

# get requirements
with open("requirements.txt") as req:
    REQUIREMENTS = req.read().strip().split("\n")

py_version_tag = '-%s.%s'.format(sys.version_info[:2])
setuptools.setup(name="packtml",
                 description="Hands-on Supervised Learning - teach a machine "
                             "to think for itself!",
                 author="Taylor G Smith",
                 author_email="taylor.smith@alkaline-ml.com",
                 packages=['packtml',
                           'packtml/clustering',
                           'packtml/decision_tree',
                           'packtml/metrics',
                           'packtml/neural_net',
                           'packtml/recommendation',
                           'packtml/regression',
                           'packtml/utils'],
                 zip_safe=False,
                 include_package_data=True,
                 install_requires=REQUIREMENTS,
                 package_data={"packtml": ["*"]},
                 python_requires='>=3.5, <4',
                 version=VERSION)
