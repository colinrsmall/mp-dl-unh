#!/usr/bin/env bash

cd "$(dirname "$0")"

if [[ "$OSTYPE" == "linux-gnu" ]]; then # assume sript is run at the SDC
    export PATH=/tools/anaconda3.2018.12/bin
    export PYTHONPATH=/tools/anaconda3.2018.12/lib/python3.6/site-packages
    python3 processor.py $1 $2 mms1 >> ~/logs/mp-dl.unh.log

elif [[ "$OSTYPE" == "darwin19" ]]; then # assume script is run on Colin Small's laptop
    python3 processor.py $1 $2 mms1 >> mp-dl-unh_log.log
fi

#function finally {
#
#}
#
#trap finally EXIT