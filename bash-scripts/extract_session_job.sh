#!/bin/bash
MULTIDAY_FOLDER=$1
DATA_FOLDER=$2
BIN_FOLDER=$3
DATA_PATH=$4
echo "multiday_folder: $MULTIDAY_FOLDER"
echo "data_folder: $DATA_FOLDER"
echo "bin_folder: $BIN_FOLDER"
echo "data_path: $DATA_PATH"
conda init bash
conda activate suite2p
echo $CONDA_DEFAULT_ENV
result=$(python <<EOF
from multiday_suite2p.cluster.extract import extract_traces_session
extract_traces_session("$MULTIDAY_FOLDER","$DATA_FOLDER","$BIN_FOLDER","$DATA_PATH")
EOF
)
echo $result
