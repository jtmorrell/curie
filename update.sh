#!/bin/bash

##### Manual PyPI upload (fallback; releases normally publish from CI on a v* tag)
python3 -m build
python3 -m twine upload dist/*
rm -r dist
