#!/bin/bash
cd /data/asr_bench; export HF_HOME=/data/asr_bench/hf
M=nvidia/nemotron-speech-streaming-en-0.6b
run(){ echo "== $*"; ./venv/bin/python -I scripts/bench_nemo.py "$@" 2>&1 | tail -1; }
run $M 6 0; run $M 1 0; run $M 13 0; run $M 0 0
run $M 6 1; run $M 1 1
run $M 1 0 1; run $M 6 0 1
PAD_S=1 run $M 6 0; PAD_S=1 run $M 1 0
echo SWEEPDONE
