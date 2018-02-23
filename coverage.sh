#!/bin/bash
coverage run --source serveit -a -m tests.test_utils
coverage run --source serveit -a -m tests.sklearn.test_prediction_server
coverage run --source serveit -a -m tests.sklearn.test_sklearn_server
coverage xml --omit serveit/config.py
python-codacy-coverage -r coverage.xml
