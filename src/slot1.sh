#!/usr/bin/env bash


export PYTHONPATH=$(dirname "$(readlink -f "$0")")
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python3 baseline/slot1_entity_attribute.py -conf_name=semeval_local.conf > slot1_log.txt 2>&1
