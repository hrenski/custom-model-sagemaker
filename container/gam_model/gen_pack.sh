#! /bin/bash

rm -rf *.egg-info/ build/ dist/
pip uninstall -y gam-model
python3 setup.py bdist_wheel
pip install dist/gam_model-0.0.1-py3-none-any.whl