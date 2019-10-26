#!/usr/bin/env bash

cd "$(dirname "$0")"

if [[ "$OSTYPE" == "linux-gnu" ]]; then # assume sript is run at the SDC
    set -o pipefail
    export PATH=/tools/anaconda3.2018.12/bin:$PATH​​
    export PYTHONPATH=/tools/anaconda3.2018.12/lib/python3.6/site-packages
    python3 processor.py $1 $2 $3 2>&1 | tee -a ~/logs/mp-dl-unh.log
    exit
elif [[ "$OSTYPE" == "darwin19" ]]; then # assume script is run on Colin Small's laptop
    set -o pipefail
    python3 processor.py $1 $2 $3 2>&1 | tee -a mp-dl-unh_log.log
    exit $?
fi