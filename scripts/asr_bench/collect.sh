#!/bin/bash
# snapshot the rolling last_utts ring (keeps newest 20) so we accumulate a test set. read-only on the live side.
while true; do cp -n /home/iris/IRIS/state/voiceid/last_utts/*.wav /data/asr_bench/clips_raw/ 2>/dev/null; sleep 20; done
