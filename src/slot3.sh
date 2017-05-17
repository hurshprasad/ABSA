#!/usr/bin/env bash


export PYTHONPATH=$(dirname "$(readlink -f "$0")")
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/hursh/submission/src

python3 baseline/slot3_attention_sentiment.py -conf_name=semeval_base.conf > slot3_log.txt 2>&1
