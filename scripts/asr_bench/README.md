# asr_bench — offline streaming-ASR benchmark (ears candidates)

Offline harness used 2026-11-02 to compare NVIDIA Nemotron Speech Streaming en-0.6B
(cache-aware RNN-T, NeMo) and Parakeet-Realtime-EOU-120M against faster-whisper
small / large-v3-turbo on real utterances saved by the ears (`state/voiceid/last_utts`).
Nothing here touches the live ears, mouth, or host.

Layout on the server (NOT in git — audio, references, and results are personal data):

    /data/asr_bench/venv/        isolated python3.11 venv: torch 2.10 cu126 (has sm_70), nemo_toolkit 3.0.0,
                                 faster-whisper 1.2.1 + ctranslate2 4.8.2 (= live), jiwer, whisper_normalizer
    /data/asr_bench/clips/       frozen test set (16 kHz mono wavs copied from the last_utts ring)
    /data/asr_bench/ref/refs.json  adjudicated references; `{a|b|}` = acceptable alternatives
    /data/asr_bench/results/     one JSON per model/config + logs

Scripts (copy of what ran):

    bench_whisper.py MODEL BEAM      decode wall time after the full clip (= whisper final latency), live params
    bench_nemo.py MODEL RC AMP [GRAPHS]
                                     cache-aware streaming on a SIMULATED real-time clock; RC = right context
                                     in 80 ms frames (0→80 ms, 1→160, 6→560, 13→1120 ms chunks); PAD_S=1 env
                                     appends 1 s of zeros (the mic keeps running after he stops)
    score.py                         WER (Whisper English normaliser, best alternative per ref)
    latency.py                       first-word lag / final-text lag / <EOU> lag vs large-v3 word timestamps
    collect.sh                       snapshots the 20-file last_utts ring so the test set can grow

ctranslate2 needs the CUDA 12 libs that ship inside the torch wheels:

    SP=venv/lib/python3.11/site-packages/nvidia
    export LD_LIBRARY_PATH=$PWD/$SP/cublas/lib:$PWD/$SP/cudnn/lib:$PWD/$SP/cuda_runtime/lib

## Live integration (2026-11-02, branch ears-stream-1102)

The bench became the ears' streaming path:

    voice/iris_stream_asr_engine.py   NeMo side: incremental log-mel (bit-identical to the whole-clip pass)
                                      + the exact CacheAwareStreamingAudioBuffer chunk/cache schedule
    voice/iris_stream_asr_server.py   the sidecar: unix socket, one session per utterance, partial + final
    voice/iris_stream_asr.py          ears-side client (stdlib + numpy), "Iris" post-fix, health counters
    scripts/server/iris-stream-asr.service   systemd --user unit (not auto-installed)
    voice/wren_voice_core.py          flag `stream_asr` in scratch/voice_control.json (off|on), see the
                                      "Streaming ASR sidecar" block

Checks that go with it (NeMo venv for the first, ears venv for the second):

    stream_replay.py BENCH_JSON [alpha] [phrases]   clips through the LIVE incremental path, 100 ms frames;
                                                    text must equal the offline bench (31/31 on 11-02)
    stream_e2e.py session|core [N]                  against a running test sidecar (IRIS_STREAM_ASR_SOCK):
                                                    real-time feed -> final latency; `core` runs the real
                                                    _capture_utterance + _transcribe_stream_turn
