#!/bin/bash

##### To upload to PyPi (will need to change old, new version numbers)
python3 setup.py sdist
python3 -m twine upload dist/*
rm -r curie.egg-info
rm -r dist

##### To copy to local package index
# rm -r ~/.local/lib/python3.10/site-packages/curie/
# mkdir ~/.local/lib/python3.10/site-packages/curie
# cp ./curie/* ~/.local/lib/python3.10/site-packages/curie/
# cp -r ./curie/data ~/.local/lib/python3.10/site-packages/curie/data/