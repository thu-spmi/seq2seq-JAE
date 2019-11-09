#!/bin/bash
#process data
PYTHONPATH=. python asdl/lang/lambda_dcs/dataset.py
PYTHONPATH=. python asdl/lang/py/dataset.py
#split data
PYTHONPATH=. python scripts/split_dataset.py data/atis/train.bin
PYTHONPATH=. python scripts/split_dataset.py data/django/train.bin