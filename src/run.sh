#!/bin/sh
BASEDIR=$(dirname `readlink -f $0`)

python3 ${BASEDIR}/service.py --port $PORT