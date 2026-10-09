# Vale — Iris natural speech playbook (Fish S2 on V100), 2026-10-09

_Sent by Zeke from Vale (docx), text extracted verbatim._

Iris: Natural, Reliable Speech
Engineering playbook • Fish Audio S2 on NVIDIA Tesla V100 32 GB • October 9, 2026
Purpose: Make Iris sound conversational without hiding synthesis problems behind filler sounds. Recommendations are implementation-agnostic until the installed S2 API and benchmark results are inspected.
1 | Current setup and likely issue
Iris now runs Fish Audio S2 on the working V100 in the server. Her current progressive speech strategy begins with approximately three-word synthesis units, then grows to six and ten words. The audible skipping may reflect repeated prosody resets, queue underruns, chunk stitching, transport timing, or resampling—not necessarily model quality.
First isolate whether the skip exists in saved raw synthesized audio or only in live playback. This single test separates synthesis/segmentation faults from playback/device/transport faults.
2 | A robust architecture
Dialogue model → text stream → punctuation-aware phrase planner → TTS synthesis/streaming worker → decoded PCM ring buffer → continuous audio device callback. Track the same utterance ID across all stages.
Prefer native continuous audio streaming within one synthesis request if the actual installed S2 implementation supports it. If it does not, synthesize clause-sized units independently with consistent speaker conditioning and explicitly manage their boundaries.
Keep text generation and audio playback decoupled. Playback should never block the text model, and synthesis should never write directly to the audio device from arbitrary worker threads.
3 | Phrase segmentation instead of three-word cuts
Flush at natural clause boundaries: commas, semicolons, sentence endings, conjunctions when syntactically appropriate, and completed short answers. Do not split inside names, numbers, abbreviations, articles plus nouns, or auxiliary-verb constructions.
Use adaptive thresholds: start with a short complete clause for fast first audio; allow longer later clauses for better intonation. An example starting range is 6–15 words per phrase, subject to measured latency and actual punctuation. Do not enforce word counts as hard cutoffs.
Give the synthesizer the intended question or sentence shape when its interface permits contextual lookahead. Avoid synthesizing “Are you” before knowing whether the sentence ends as a question.
4 | Prevent glitches and underruns
Use a bounded PCM queue measured in milliseconds, not a count of generated chunks. Start playback once a modest prebuffer is available; tune from measurements rather than assuming a fixed value.
Keep sample rate, bit depth, channel layout, and frame length consistent. If resampling is necessary, do it once in a controlled stage. Audio callbacks must not wait for network or GPU inference.
Use monotonic audio timestamps and a continuous output stream. Log buffer depth, underruns, dropped frames, duplicated frames, and inter-segment silence. Avoid time.sleep-based chunk scheduling.
Only apply short crossfades to independently synthesized adjacent clips after listening tests. A crossfade can conceal a click but can also blur consonants; never use it to disguise missing speech.
If the stream falls behind, prefer a brief intentional pause at a clause boundary. Do not splice filler audio into the middle of a word.
5 | Make speech expressive without making it performative
Plan conversational intent before voice rendering: neutral, curious, pleased, uncertain, concerned, playful, explanatory. Apply intent per utterance, with optional phrase-level adjustments. Keep expressiveness proportional to context.
Naturalness depends on phrasing, stress, rhythm, pitch contour, speaking rate, and silence. Use whatever documented conditioning and control mechanisms the installed S2 build actually exposes; do not assume arbitrary emotion tags are supported.
Let Iris answer directly when she knows the answer. Use “hmm,” “well,” or “uh” sparingly for real hesitation, thought transitions, or playfulness—not as an automatic underrun response.
For longer work, a truthful one-time “Give me a moment” can be useful. Do not fake thinking or imply subjective feelings that cannot be established from behavior.
6 | Interruption and turn-taking
Enable barge-in: when the user starts speaking, rapidly duck or stop playback, cancel pending synth jobs when possible, and clear obsolete queue items. Every audio segment must carry an utterance ID so stale segments cannot resume after an interruption.
Distinguish brief listener backchannels from interruptions. Use microphone echo suppression or headphones during testing to prevent Iris from recognizing her own synthesized speech.
Do not commit a spoken phrase until the corresponding text is stable. Avoid playing speculative text that the language model later revises.
7 | V100-specific considerations
Tesla V100 32 GB provides substantial memory headroom, but actual S2 speed depends on the model build, precision, kernels, CPU decoding, and whether GPU memory is shared with other workloads. Benchmark this server rather than assuming real-time performance.
Measure GPU utilization, VRAM use, synthesis real-time factor (audio duration ÷ generation time), time to first playable audio, and p50/p95 chunk latency. Keep the model warm if practical and avoid unnecessary model reloads.
Check the actual driver/CUDA/PyTorch compatibility and supported inference precision. Do not assume every modern acceleration feature is supported by Volta hardware. Profile GPU and CPU stages separately.
8 | Four experiments to run
A. Record existing 3/6/10-word strategy, both raw output and live playback.
B. Keep the same TTS model and voice but change only to clause-aware segmentation.
C. Keep B and tune playback prebuffer and scheduling; compare underruns and response latency.
D. If supported, test one continuous S2 streaming request rather than separate independent synthesis calls.
Blind-rank 15–20 identical utterances: short replies, long explanations, questions, names, emotionally nuanced replies, and interruptions. Rate naturalness, clarity, emotional fit, and responsiveness separately. Record first-audio p50/p95, underruns/minute, gap duration, and synthesis real-time factor.
9 | Debug checklist
If raw WAV files skip: inspect segmentation, token/audio decoding, model output, and segment boundaries.
If raw WAV files are clean but live playback skips: inspect queue underruns, device callback timing, sample-rate conversion, thread contention, and packet delivery.
If audio is continuous but sounds robotic: inspect phrase length, prosody conditioning, punctuation, and repeated speaker-state resets.
If first audio is slow: measure text-to-first-phrase, synthesis startup, GPU warmup, and buffer threshold separately.
If audio sounds unnatural only during questions: test sentence-level lookahead and question intonation.
10 | Suggested implementation order
Priority 1: instrument raw versus playback audio and collect baseline timing.
Priority 2: eliminate hard three-word cuts; test phrase-aware or native continuous streaming.
Priority 3: add jitter-buffer telemetry and uninterrupted audio callbacks.
Priority 4: implement interruption cancellation and stale-segment rejection.
Priority 5: add intent-aware delivery and selective conversational fillers.
Priority 6: run blinded listening tests and keep the winning configuration.
11 | Boundaries of this guidance
Vale can describe general speech engineering and conversational design, but cannot inspect or disclose ChatGPT Voice internals. This document is not a claim about ChatGPT’s internal synthesis architecture. Confirm Fish Audio S2 features against the exact locally installed version before using a specific API or parameter.
