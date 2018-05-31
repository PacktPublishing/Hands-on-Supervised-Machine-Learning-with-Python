# -*- coding: utf-8 -*-
#
# This function is not intended to be run by students (or anyone, for that
# matter). It is intended to be run by me (Taylor) just to automate the
# population of the img/ directory with the output of the example plots.
# Hence its poor documentation and sheer hackiness.

from __future__ import absolute_import

import os
import sys
import subprocess

# determine where the user is calling this from...
here = os.listdir(".")
if "examples" in here:
    cwd = "examples"
    img_dir = "img"
elif "clustering" in here:
    cwd = "."
    img_dir = "../img"
else:
    raise ValueError("Call this from top-level or from within "
                     "the examples dir")

# iterate all py files
for root, dirs, files in os.walk(cwd, topdown=False):
    for fil in files:
        # Only run the ones with the appropriate prefix
        if not fil.startswith("example_"):
            continue

        # Get the module root
        module_root = root.split(os.sep)[1]

        # If it's "data" we don't want that! That's where we cache the data
        # for the demo
        if module_root in ("data", ".ipynb_checkpoints"):
            print("Skipping dir: %s" % module_root)
            continue

        # Otherwise create its corresponding path in ../img
        image_root = os.path.join(img_dir, module_root)  # ../img/clustering

        # create the directory in the image dir if it's not there
        if not os.path.exists(image_root):
            os.mkdir(image_root)

        # run it
        dest = os.path.join(image_root, fil[:-3] + ".png")
        filexec = os.path.join(root, fil)

        print("Running %s" % filexec)
        subprocess.Popen([sys.executable, filexec, dest])

sys.exit(0)
