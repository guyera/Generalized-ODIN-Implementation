#!/bin/bash

python -u main.py --magnitudes 0 0.0014 0.0025 0.005 0.01 0.02 0.04 0.08 --epochs 300 | tee results-300-epochs-he-bs-64-tr-0.15.txt &
