#!/bin/bash
#
# This script creates documentation for parent and sub folders/modules. The script uses the sphinx python package.
# If it is not present use one of the following commands to install it:
#
# conda install sphinx
# pip install sphinx

# Target directory for the documentation.
#
work_dir="_sphinx"
html_dir="html"

# Create documentation.
#
echo "[1/3] Creating documentation."
sphinx-apidoc .. --full --separate --output-dir=./${work_dir} --doc-project="DigitalPathology" --doc-author="Computational Pathology Group" --doc-version="1.0"

# Build HTML structure.
#
echo "[2/3] Generating HTML."
make --directory=./${work_dir} html

# Clean up the intermediate files.
#
echo "[3/3] Cleaning up."
cp --recursive ./${work_dir}/_build/html ./${html_dir}
rm --recursive --force ./${work_dir}
