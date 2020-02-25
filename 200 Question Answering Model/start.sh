#!/usr/bin/env bash

DATA_DIR=./data

mkdir -p $DATA_DIR
python3.6 preprocessing/download_wordvecs.py --download_dir=$DATA_DIR
python3.6 preprocessing/squad_preprocess.py --data_dir $DATA_DIR