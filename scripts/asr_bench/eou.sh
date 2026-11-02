#!/bin/bash
cd /data/asr_bench; export HF_HOME=/data/asr_bench/hf
until grep -q SWEEPDONE results/sweep.log; do sleep 10; done
./venv/bin/python -I scripts/bench_nemo.py nvidia/parakeet_realtime_eou_120m-v1 1 0 > results/log_eou_rc1.txt 2>&1
PAD_S=1 ./venv/bin/python -I scripts/bench_nemo.py nvidia/parakeet_realtime_eou_120m-v1 1 0 > results/log_eou_rc1_pad.txt 2>&1
echo EOUDONE >> results/log_eou_rc1_pad.txt
