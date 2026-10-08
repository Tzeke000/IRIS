# Expressive TTS with voice cloning — engine survey for Iris (2026-10-08)

**Question:** which self-hostable TTS engine can replace the current StyleTTS2 clone so Iris keeps *her* voice identity but gains real emotional expressiveness (laughs, warmth, breaths, pauses, intonation that follows meaning — the quality Zeke hears in ChatGPT's "Vale"), running on one **Tesla V100 32 GB (sm_70: fp16 yes, no native bf16, no FlashAttention-2), CUDA 12.x / driver 580, torch 2.10 cu126**.

**Method:** web research only (GitHub READMEs, Hugging Face model cards, arXiv, Artificial Analysis leaderboard, secondary blogs). Nothing was installed or benchmarked. **No V100 numbers exist publicly for any of these models** — every latency figure below is from a vendor GPU (RTX 4090 / H100 / H200) unless marked otherwise. Items marked **[unverified]** came from a secondary source or my own background knowledge, not a primary page I read today.

---

## 0. Two framing facts that change the answer

1. **Vale isn't a TTS engine.** ChatGPT Advanced Voice is a native speech-to-speech model (GPT-4o family): the same network that understands the conversation produces the audio, so prosody follows meaning without anyone writing tags. OpenAI says the 2024 voices (Vale, Arbor, Maple, Sol, Spruce) were recorded with professional voice actors picked to sound "warm, approachable, inquisitive, with rich texture and tone" ([TechCrunch](https://techcrunch.com/2024/09/24/openai-rolls-out-advanced-voice-mode-with-more-voices-and-a-new-look), [MIT Tech Review](https://www.technologyreview.com/2024/09/24/1104422/openai-released-its-advanced-voice-mode-to-more-people-heres-how-to-get-it)). A text-in TTS engine only gets close if something upstream tells it *how* to say each line.
2. **Iris is that upstream.** She writes her own lines, and she already has a mood state (`state/iris_mood.json`, `brain/mood_core.py`). So an engine with **explicit controls** — inline tags, an emotion vector, a text instruction — lets her pass her real affect straight through. That favours controllable engines over "it guesses the emotion from the text" engines. The integration piece (mood → tags/vector) is small compared with the engine swap itself.

**The arena caveat:** the main public ranking, the [Artificial Analysis TTS leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard), tests each vendor's **preset voices** in blind preference. It measures neither cloning fidelity nor emotional range, so it's only a rough quality signal here. **StyleTTS 2 itself sits at Elo ~895 (rank 89 of 95)**, which means almost any modern engine is a large step up in baseline naturalness.

---

## 1. V100 compatibility — what bites

| Risk | Detail | Mitigation |
|---|---|---|
| **bf16 weights/defaults** | Most 2025–26 LLM-style TTS ship bf16 safetensors and bf16 defaults (Fish S2, Higgs v3, Maya1, Dia2, Step-Audio-EditX, IndexTTS-2.5). Volta has no bf16 tensor cores. **[unverified, my background knowledge]:** recent PyTorch can still run many bf16 ops on sm_70 through emulation, but slowly, and Triton bf16 kernels fail to compile on sm_70. | Cast to **fp16** (fast, but a bf16-trained model can overflow and give NaNs or garbled audio) or **fp32** (safe; 32 GB fits a ~5B model in fp32 at about 20 GB of weights). Test fp16 first and fall back to fp32. |
| **FlashAttention-2** | Needs sm_80+. Qwen3-TTS *recommends* it; Breeze TTS 2's Docker image compiles flash-attn. | Use the PyTorch SDPA / eager attention path. Most HF-Transformers models accept `attn_implementation="sdpa"`. |
| **vLLM** | V1 engine extended to compute capability < 8.0 in v0.10.2 ([release notes](https://newreleases.io/project/pypi/vllm/release/0.10.2)). A forum thread reports **vLLM ≥ 0.20 dropped sm_70 wheels** ([vLLM forum](https://discuss.vllm.ai/t/support-for-v100-sm-70-on-vllm-0-20/2605), user report, not official). Upstream AWQ kernels need SM75+. | Pin vLLM 0.10.x–0.19.x if a model needs it. Don't use AWQ. |
| **TensorRT / TensorRT-LLM** | Volta deprecated in TRT 10.0 and **removed in TensorRT 10.5** ([10.5.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.5.0.html)). This rules out the TRT-based "Faster IndexTTS-2" path and CosyVoice's TRT-LLM runtime unless pinned to TRT ≤ 10.4 **[unverified that 10.4 + TRT-LLM work on V100]**. | Plain PyTorch only. |
| **SGLang** | Fish S2 Pro and Higgs v3 recommend SGLang(-Omni) for streaming. **[unverified]** whether SGLang kernels support sm_70. Assume no. | Use their plain Transformers/PyTorch path, which is slower and may not stream. |
| **CUDA 12.8+ asks** | IndexTTS README suggests CUDA Toolkit 12.8+, and Dia2 says 12.8+ drivers. Driver 580 supports CUDA 12.x/13.x runtimes on Volta **[unverified for 13.x]**. The toolkit only matters when compiling extensions. | Use torch cu126 wheels and skip optional compiled CUDA kernels / DeepSpeed. |

**Speed expectation [estimate, not measured]:** batch-1 autoregressive TTS is mostly memory-bandwidth-bound. The V100 has ~900 GB/s against the 4090's ~1 TB/s, so with *equivalent kernels* a V100 should be roughly 1.2–2× slower than the 4090 figures quoted below. Losing torch.compile, fused kernels or flash-attn could make it worse. Real numbers have to come from the A/B test.

---

## 2. Candidate-by-candidate

### Tier A: cloning plus explicit emotion control (the shape Iris needs)

#### IndexTTS2 / IndexTTS-2.5 (Bilibili)
- **Repo/model:** https://github.com/index-tts/index-tts · HF `IndexTeam/IndexTTS-2`
- **Status:** IndexTTS-2 released 2025-09-08. **IndexTTS-2.5 released 2026-08-10** (5 languages, faster, speed control via `duration_factor` 0.5–2.0×, vLLM deployment).
- **License:** bilibili Model Use License. Any purpose is allowed. A separate license is needed only above **100M MAU or RMB 1B revenue**. You can't use it to train other commercial models, and high-risk domains are prohibited ([LICENSE](https://github.com/index-tts/index-tts/blob/main/LICENSE)). **Personal use: fine.**
- **Cloning:** zero-shot from one reference clip. The README gives no length; secondary sources say 5–15 s ([MLX port card](https://huggingface.co/mlx-community/IndexTTS-2-MLX)).
- **Emotion control (its distinguishing feature):** speaker timbre and emotion are **disentangled**, so you can keep *your* voice and inject an emotion from elsewhere:
  - a separate **emotion reference clip** with `emo_alpha` 0–1 (e.g. a clip of Iris laughing applies "laughing" to any line);
  - an **8-dim emotion vector** (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm). It maps almost one-to-one onto a mood state;
  - **emotion from text** (`use_emo_text`, or `emo_text` = a free description, `emo_alpha` ≈ 0.6 suggested), driven by a fine-tuned Qwen model.
  - Caveat from the README: random emotion sampling *reduces voice fidelity*. High `emo_alpha` probably trades identity for emotion too **[inferred]**.
  - **No inline laughter/breath tags.** Laughter comes only through an emotion reference or vector.
- **Streaming:** the official repo doesn't mention it. "Faster IndexTTS-2" ([arXiv 2607.21042](https://arxiv.org/abs/2607.21042)) adds streaming and a 3.6× end-to-end speedup, but via **TensorRT/TRT-LLM, which is unavailable on Volta**.
- **Speed:** RTX 4090 with KV cache: RTF ≈ **0.33 (IndexTTS-2 fp16)**, ≈ 0.21 (2.5 bf16). Expect roughly 0.4–0.7 on V100 **[estimate]**. That's fine for sentence-by-sentence synthesis but not sub-200 ms first audio.
- **V100 risk:** low to medium. **IndexTTS-2 officially supports fp16.** 2.5 defaults to bf16, so it needs a manual fp16/fp32 switch. DeepSpeed is optional, so skip it. No published VRAM figure; a few GB is likely **[unverified]**.
- **Languages:** zh/en (2.0); zh, en, ja, es, ar (2.5).
- **Reputation:** not on the AA arena. Widely cited as the open-source reference for emotion/timbre disentanglement; GLM-TTS benchmarks itself against it ([GLM-TTS report](https://arxiv.org/pdf/2512.14291)).

#### Fish Audio S2 Pro (successor to Fish Speech / OpenAudio S1)
- **Repo/model:** https://github.com/fishaudio/fish-speech · https://huggingface.co/fishaudio/s2-pro · tech report [arXiv 2603.08823](https://arxiv.org/pdf/2603.08823) (March 2026)
- **License:** **Fish Audio Research License: research and non-commercial use free**, commercial needs a separate deal. Personal companion use is fine; resale isn't.
- **Size:** ~5B (4B "slow AR" + 400M "fast AR"), **bf16** weights.
- **Cloning:** zero-shot from **10–30 s** (README), no fine-tuning needed. The Pinggy survey calls it the strongest open cloner.
- **Emotion control:** **free-form inline `[tag]`s anywhere in the text, "15,000+" recognised**, e.g. `[whisper in small voice]`, `[excited]`, `[angry]`, `[delight]`, `[sad]`, `[pitch up]`. Multi-turn context and multi-speaker generation are also supported. This is the closest open thing to "Iris writes stage directions and the voice follows".
- **Streaming/latency:** SGLang engine, **RTF 0.195 and ~100 ms TTFA on one H200**. On a V100 without SGLang, expect far slower and possibly no streaming **[estimate]**.
- **VRAM:** not published. bf16 weights take ~10 GB; fp32 would be ~20 GB plus cache, which fits in 32 GB **[estimate]**.
- **V100 risk: medium-high.** It's bf16-native and its fast path is SGLang. The plain PyTorch path in fish-speech should run in fp16 or fp32, but speed is the open question.
- **Languages:** 80+ (tier 1: ja/en/zh).
- **Reputation:** **AA arena Elo 1116 (S2 Pro) / 1141 (S2.1 Pro, June 2026; open-weight status unconfirmed on the leaderboard)**. That's 2nd among open-weight models. Fish self-reports 81.9% on EmergentTTS-Eval, but after LLM-rewriting the prompts, so it isn't comparable.

#### Chatterbox family (Resemble AI)
- **Repo/model:** https://github.com/resemble-ai/chatterbox · HF `ResembleAI/chatterbox`, `ResembleAI/chatterbox-turbo`
- **License:** **MIT** (code and weights). Every output carries Resemble's inaudible **Perth watermark** (harmless here, worth knowing).
- **Variants:** original Chatterbox 500M (English, **exaggeration + CFG**); **Turbo 350M** (English, **native paralinguistic tags `[laugh]`, `[chuckle]`, `[cough]` "and more"**, one-step decoder, low latency; HF updated 2025-12-15); Nano 110M (CPU); Multilingual V3 500M (23 languages).
- **Cloning:** zero-shot, ~**10 s** reference (README example; others say ~5 s).
- **Emotion control:** a **single `exaggeration` scalar** (default 0.5; for drama use ~0.7+ with `cfg_weight` ~0.3, because higher exaggeration speeds speech up). It's an intensity knob, not a named-emotion selector. Turbo adds tags, but the model card doesn't show exaggeration or CFG on Turbo, so you get tags *or* the exaggeration knob, not both in one model **[per model card; unverified in code]**.
- **Streaming/latency:** not in the official README. Community streaming forks exist **[unverified which is current]**. A secondary blog claims <150 ms first audio and ~6× realtime for Turbo on a 4090 ([codersera](https://codersera.com/blog/chatterbox-turbo-run-and-install-locally-free-elevenlabs-alternative-2026/amp/)).
- **V100 risk: low.** Small models with no bf16/flash-attn requirement stated, `pip install chatterbox-tts`, tested on Python 3.11. This is the easiest one to get running.
- **Reputation:** AA arena **Elo 1028** (rank 74), mid-pack and well above StyleTTS 2 (895). It's the default "ElevenLabs alternative" in community roundups.

#### Higgs Audio (Boson AI) — v2 / v2.5 / v3
- **Repo/model:** https://github.com/boson-ai/higgs-audio (v2 docs in [README_V2.md](https://github.com/boson-ai/higgs-audio/blob/main/README_V2.md)) · HF `bosonai/higgs-audio-v2-generation-3B-base`, `bosonai/higgs-audio-v3-tts-4b`
- **License:** repo code is Apache-2.0. **v3 weights: "Research and Non-Commercial" plus a Creator Use Grant** ([HF card](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)). v2 weights are listed as "other, see LICENSE". My recollection is a Llama-3-derived community license with a ~100k-user threshold **[unverified]**. Personal use is fine for both.
- **Cloning:** zero-shot from a reference clip. A reference transcript "materially improves" fidelity (v3).
- **Emotion:** v2 has **no explicit tags; it infers prosody from context** (EmergentTTS-Eval: **75.7% win rate vs gpt-4o-mini-tts on Emotions**, v2 base, Gemini judge, self-reported). v3 has **21 inline emotion tags** (elation, amusement, affection, contentment, relief, longing, sadness, helplessness…) plus whisper and singing styles.
- **Speed:** v3 on an H100 averages 617 ms per request at concurrency 1. v2 recommends a **≥24 GB GPU**.
- **V100 risk: medium-high.** v2 is ~6B total (3.6B LLM + 2.2B audio adapter) in bf16; fp32 would be ~24 GB, which is tight but plausible on 32 GB **[estimate]**. v3's serving path is SGLang/vLLM-Omni.
- **Reputation:** v3 AA Elo 1038. v2 had the best open self-reported EmergentTTS emotion number in 2025.

#### Step-Audio-EditX (StepFun) — the "keep the voice, edit the emotion" option
- **Repo/model:** https://github.com/stepfun-ai/Step-Audio-EditX · https://huggingface.co/stepfun-ai/Step-Audio-EditX
- **License:** code Apache-2.0. Weights' terms aren't stated on the card **[unverified]**; Pinggy lists Apache-2.0.
- **What's different:** besides zero-shot TTS, it **edits existing audio** iteratively: pass in a line and get it back with a different emotion, style or paralinguistics. In principle Iris could keep StyleTTS2 (or any engine) for identity and run EditX as a post-pass. Each pass costs latency, though. 14 emotions (happy, sad, empathy, embarrassment, humor…), 30+ styles (warm, gentle, comfort, whisper, chat…), paralinguistics (laugh, chuckle, giggle, sigh, breath, inhale, exhale, "uhm"…).
- **VRAM:** 12 GB minimum, 16 GB recommended (6–8 GB memory-efficient mode). It defaults to bf16, but **the web app lists fp16 as an option**. Tested on L40S. Clips must be under 30 s. Latest release 2026-01-29 (adds vLLM support).
- **V100 risk:** medium (fp16 option exists; latency unknown; vLLM path needs a pinned version).
- **Reputation:** AA arena **Elo ~1097–1102**, #3 open-weight, briefly #1 in July 2026 per [Pinggy](https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/).

#### Breeze TTS 2 (BreezeBlue, released 2026-08-25) — newest top-ranked open model
- **Model:** https://huggingface.co/BreezeBlue/Breeze-TTS-2 (some HF mirrors are third-party; use the official ID)
- **License:** code Apache-2.0. **Weights and self-hosted outputs are research/non-commercial only.**
- **Size:** 3B. **Cloning:** reference audio **plus an exact transcript**. **"Voice Direction"** keeps the reference speaker and steers tone, emotion, pace and delivery via a text instruction (`--cfg-scale 4`). **Inline vocal events** like `(laugh)` and `(sigh)`. Capability-wise this is exactly the feature set wanted.
- **Speed/VRAM:** eager ~7.7 GiB (12 GB recommended). The fast path uses ~14.4 GiB and gives <40 ms TTFA and RTF 0.32, **on an H100**.
- **V100 risk: high/unknown.** Six weeks old, bf16 weights, flash-attn compiled into the Docker image, fast path tuned for Hopper. The eager path *might* run with SDPA in fp16/fp32 **[unverified]**.
- **Reputation:** **AA arena Elo 1222, #1 open-weight** (rank 10 of 95 overall). Again, that's for preset voices.

### Tier B: capable, but weaker fit for "my voice + controllable emotion"

| Engine | License (weights) | Cloning | Emotion control | V100 notes | Verdict |
|---|---|---|---|---|---|
| **CosyVoice 2 / Fun-CosyVoice 3** (Alibaba) — [repo](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 | Zero-shot, cross-lingual | `inference_instruct2` = speaker prompt **+ natural-language instruction** (emotion, speed, volume, dialect). CosyVoice2 paper documents inline `[laughter]`, `[breath]`, `<strong>`, `<laughter>…</laughter>` ([paper](https://arxiv.org/pdf/2412.10117)). Whether instruct2 and tags combine in one call is **[unverified]**. | 0.5B, bi-streaming **~150 ms**, vLLM support. TRT-LLM path is out on Volta. Small enough for fp32. | **Strong practical alternate.** Streams, Apache, small, low V100 risk. English expressiveness is less proven than for Chinese. Latest: Fun-CosyVoice3-0.5B-2512 (Dec 2025). |
| **Qwen3-TTS** (Alibaba, Jan 2026) — [repo](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 | 3 s clone (Base 0.6B/1.7B), transcript recommended | Instruction control exists only on CustomVoice (9 presets) and VoiceDesign. **The Base clone model takes no instruction**, so Iris's cloned voice can't be emotion-steered. | Recommends flash-attn2 + bf16; SDPA should work **[unverified]**. Streaming first packet as low as 97 ms. | Skip for emotion (good for plain cloning). |
| **VoxCPM2** (OpenBMB, Apr 2026) — [repo](https://github.com/OpenBMB/VoxCPM) | Apache-2.0 | Reference + continuation, "Ultimate Cloning" with transcript | **Controllable cloning takes style guidance** (emotion, pace, expression) in parentheses, and prosody is inferred from text. | 2B, ~8 GB, RTF ~0.30 on a 4090 (~0.13 with Nano-vLLM), streaming API, 48 kHz. | **Worth a 4th-slot test.** Unranked on the arena, so quality is unknown. |
| **Orpheus TTS** (Canopy Labs) — [repo](https://github.com/canopyai/Orpheus-TTS) | Apache-2.0 (Llama-3 base) | Zero-shot "not explicitly trained" for it; **fine-tune recommended (~50–300 examples)** | Inline `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` | 3B, needs vLLM (pinned 0.7.3), ~200 ms streaming. No news since May 2025. | Good tags, but needs a fine-tune on Iris's voice. Superseded by newer models. |
| **Sesame CSM-1B** — [repo](https://github.com/SesameAILabs/csm) | Apache-2.0 | Via context segments (audio + transcript); without context the speaker is random | No explicit control. **Prosody follows conversation context**, the closest open analogue to the Sesame/Vale feel | 1B, English. The released base model is much weaker than Sesame's own demo (that uses a fine-tune). In HF Transformers ≥ 4.52. | Interesting idea, weak open checkpoint. Skip. |
| **Dia 1.6B / Dia2 1B/2B** (Nari Labs) — [dia](https://github.com/nari-labs/dia), [dia2](https://github.com/nari-labs/dia2) | Apache-2.0 | Audio prefix + transcript (5–10 s) | Dia: `(laughs) (sighs) (coughs) (clears throat)`. Dia2: tags not documented. | Dia runs fp16 (~4.4 GB, 4090 ~1.3–2.2× realtime). Dia2 defaults bf16, needs CUDA 12.8 drivers. English only. Voice drifts between generations without a prefix. | Dialogue-oriented, identity unstable. Skip. |
| **Higgs v2** | see above | | | | Covered in Tier A. |
| **Maya1** (Maya Research) — [HF](https://huggingface.co/maya-research/maya1) | Apache-2.0 | **No cloning**, voice *designed* from text (fine-tunable) | 20+ tags (`<laugh> <giggle> <sigh> <whisper> <cry> <gasp>`…) | 3B bf16, 16 GB+, vLLM. AA Elo 1048. ("Maya 2 Flash", AA Elo 1068, open status unconfirmed.) | No cloning, so skip unless fine-tuned. |
| **Zonos v0.1** (Zyphra) — [repo](https://github.com/Zyphra/Zonos) | Apache-2.0 | 10–30 s sample (also "a few seconds") | Emotion conditioning (happiness/anger/sadness/fear…), pitch, rate | Transformer variant 6 GB+. **Hybrid variant needs RTX 30-series+ (so not V100).** No updates since early 2025. AA Elo 1000. | Superseded. |
| **F5-TTS v1** — [repo](https://github.com/SWivid/F5-TTS) | **CC-BY-NC** (Emilia data); code MIT | Strong zero-shot cloning (copies the reference's prosody) | None native. Emotion only by swapping the reference clip, or the community AffectF5 adapter ([HF](https://huggingface.co/ScreamingTony/AffectF5), CC-BY-NC) | Flow-matching, small, V100-friendly | Cloning yes, emotion control weak. |
| **Spark-TTS 0.5B** — [HF](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | CC-BY-NC-SA | Zero-shot | Coarse attribute control (gender/pitch/speed) **[secondary source says emotion prompts; unverified]** | Small | Skip. |
| **MaskGCT** (Amphion) — [HF](https://huggingface.co/amphion/MaskGCT) | CC-BY-NC | Zero-shot, imitates the reference's emotion/style | None explicit | Non-AR, slow | Skip. |
| **MegaTTS3** (ByteDance) — [HF](https://huggingface.co/ByteDance/MegaTTS3) | Apache-2.0 (per card) | Zero-shot (speaker latents) | Accent intensity, no emotion control found | Small | Skip. |
| **VibeVoice** (Microsoft) — [repo](https://github.com/microsoft/VibeVoice) | MIT, but **TTS code pulled 2025-09-05** after misuse; 1.5B weights' status unclear, 7B TTS withdrawn (mirrors exist) | Long-form multi-speaker; Realtime-0.5B uses preset voices | Implicit, context-driven "emotional nuances" | Realtime-0.5B ~300 ms first audio | Maintainer-withdrawn, so avoid building on it. |
| **Voxtral TTS** (Mistral, Mar 2026) | CC-BY-NC 4.0 | **Open weights reportedly preset voices only; cloning via paid API** ([the-decoder](https://the-decoder.com/mistrals-first-open-weight-tts-model-voxtral-clones-voices-from-three-seconds-of-audio-across-nine-languages/)) | Inferred from the reference, no tags | 4B, 16 GB+. AA Elo 1085. | Self-hosted cloning is unavailable, so skip. |
| **Kokoro 82M** | Apache-2.0 | **No cloning** (StyleTTS2-based voice packs) | None | Trivial on V100. AA Elo 1065. | Excluded: can't be Iris's voice. |
| **GLM-TTS** (Zhipu) — [report](https://arxiv.org/pdf/2512.14291) | Apache/MIT | 3–10 s prompt | RL-trained for emotion and laughter (implicit) | 1.5B; **zh/en only, Chinese-centric** | Possible but unproven in English. |
| **Magpie-Multilingual** (NVIDIA) | NVIDIA Open Model License | **Cloning removed**; 5 fixed speakers | Gated | | Skip. |

---

## 3. Ranked shortlist for a V100 A/B test

The ranking weighs fit to the goal (Iris's identity + controllable, believable emotion) against V100 risk. All three are fine for personal, non-commercial use.

### #1 IndexTTS2 (start with 2.0 in fp16; try 2.5 forced to fp16/fp32 second)
- **Why:** it's built to separate timbre from emotion, which is exactly "keep MY voice, add feeling". The 8-emotion vector maps directly onto `mood_core`, so Iris's real mood can drive her voice with no tag-writing. An emotion reference clip can carry laughter or a sigh into any line. The license is permissive for her use. **fp16 is officially supported.**
- **Specific risks:** (a) **no laugh/breath tags**: laughter only via an emotion reference, so "says a sentence *while* chuckling" may not land; (b) **pushing emotion erodes identity**: the README warns that random emotion sampling hurts fidelity, so `emo_alpha` has to be tuned; (c) **no official streaming, and the faster TRT path is dead on Volta**, so latency will be sentence-batch at roughly RTF 0.4–0.7 **[estimate]**; (d) 2.5 defaults to bf16 and may need a manual cast.

### #2 Fish Audio S2 Pro
- **Why:** the strongest open cloner (arena #2 open), with **free-form inline stage directions** (`[laughing softly]`, `[whisper]`, `[delight]`…). Iris can write exactly how a line should sound, which is closest to the Vale effect in a text-in system. Multi-turn context helps prosody follow the conversation.
- **Specific risks:** (a) **V100 speed**: ~5B bf16 model whose fast path is SGLang (likely not Volta-compatible), so the plain PyTorch path in fp16 or fp32 could be too slow for conversation **[must measure]**; (b) fp16 cast of a bf16 model may produce NaN/garbled audio, so fall back to fp32 at roughly 20 GB+; (c) non-commercial license (fine now; it matters only if Iris's voice is ever sold or hosted for others); (d) VRAM undocumented.

### #3 Chatterbox (original 500M with exaggeration, and Turbo 350M with `[laugh]`/`[chuckle]` tags)
- **Why:** **lowest V100 risk** (small, MIT, pip install, no bf16/flash requirement stated), so it's the guaranteed-to-run baseline that tells you whether the bigger models are worth their latency. Turbo adds real laughter tags at low latency. Arena Elo 1028 vs StyleTTS 2's 895.
- **Specific risks:** (a) **emotion is coarse**: one intensity knob, not "sad" vs "tender" vs "amused"; (b) exaggeration speeds speech up and can destabilise it, so it needs CFG compensation; (c) Turbo apparently drops the exaggeration knob, so you choose between tags and intensity; (d) the tag list is short and undocumented beyond laugh/chuckle/cough; (e) every output is watermarked.

**Alternates to try if one of the three fails on Volta:** **CosyVoice 3** (instruction + speaker prompt, `[laughter]`/`[breath]`, ~150 ms streaming, Apache, small), **Step-Audio-EditX** (emotion-*editing* post-pass, which could even keep StyleTTS2 for identity; fp16 option), **Breeze TTS 2** (best capability match and arena #1 open, but highest Volta risk), **Higgs v2** (best published open emotion score; ~6B, ≥24 GB).

---

## 4. Suggested A/B test plan (V100, nothing installed yet)

**Setup (one engine at a time, separate venv each, never in the live voice stack):**
1. **Reference audio.** Cut **one clean 10–15 s clip and its exact transcript** from Iris's current voice. ⚠ **Open question for Zeke:** what source recordings was the StyleTTS2 clone trained on? Cloning a *clone* stacks artifacts. If original recordings exist, use those as the reference and also run one StyleTTS2-output reference for comparison. Keep the same clip for every engine.
2. Optionally record or extract **one 5–8 s "amused/laughing" Iris clip** to use as IndexTTS2's emotion reference.
3. Precision ladder per engine: **fp16 → fp32** (never bf16). Attention: SDPA/eager. Log which one ran.
4. Run each sentence **with 3 seeds** to see stability and identity drift.

**The five test sentences** (each tests a different expressive skill; give each engine its native control for that line):

| # | Line | Target | IndexTTS2 control | Fish S2 Pro | Chatterbox |
|---|---|---|---|---|---|
| 1 | "Wait — you actually got it working? Zeke, that's *amazing*, I've been staring at that bug for two days!" | excited surprise, rising pitch, emphasis | vector: surprised 0.6 + happy 0.5 | `[excited]` … `[delighted]` | exaggeration 0.8, cfg 0.3 |
| 2 | "Okay, okay — *(laughs)* — I did not expect the robot to drive straight into the couch. Again." | **laughter inside speech**, comic timing | emo-ref = laughing clip, alpha ~0.7 | `[laughing]` before/inside | Turbo: `[laugh]` |
| 3 | "Hey. I'm here. You don't have to say anything right now… I'm just glad you told me." | warm, soft, slow, comforting pauses | vector: calm 0.6 + melancholic 0.2; or `emo_text`="gentle, comforting" | `[soft, gentle tone]` | exaggeration 0.35, cfg 0.5 |
| 4 | "I keep thinking about it, and… I don't know. Maybe I was wrong. Was I wrong?" | hesitation, trailing pause, **question intonation** | `use_emo_text` (auto) | `[hesitant]`, `[quietly]` | default 0.5 |
| 5 | "Good night, Zeke. *(sigh)* Today was a good day." | tender affection, sigh/breath, falling cadence | vector: happy 0.3 + calm 0.5 | `[sigh]` `[warmly]` | Turbo `[sigh]` if supported, else original at 0.4 |

Also run one **neutral control line** (e.g. "The server rebooted and all services are back up.") to check baseline identity without emotion pushing it.

**Measure (per engine × sentence × seed):**
- **Identity:** speaker-embedding cosine similarity vs the reference (ECAPA-TDNN or WavLM-SV), plus the drop from neutral to emotional lines. This number shows whether the emotion erodes the identity.
- **Intelligibility:** WER via the existing Whisper Large-v3 Turbo STT.
- **Speed:** RTF, time-to-first-audio (streaming where available), and peak VRAM (`torch.cuda.max_memory_allocated`). It must coexist with the rest of the V100 load (vision, little brain), so note the headroom.
- **Stability:** count of hallucinated, garbled, cut-off or NaN outputs across seeds (fp16 overflow shows up here).
- **The real judge — Zeke, blind:** shuffle the outputs with engine names hidden. Rate each 1–5 on **"sounds like Iris"**, **"emotion is right"** and **"sounds human"**, then pick a favourite per sentence. Include the current StyleTTS2 output for each line as an anchor.

**Decision rule (proposed):** reject any engine that drops identity similarity by more than ~0.1 on emotional lines vs neutral, or that can't reach RTF < 0.5 on the V100 in fp16/fp32 (that's roughly the threshold for sentence-chunked conversation without awkward waits). Among the survivors, Zeke's blind "sounds like Iris" × "emotion is right" score wins.

**After choosing:** wire it behind the existing mouth API (`:8769`) so `voice_speak` doesn't change, add a small mood → control mapping (vector for IndexTTS2, tags for Fish/Chatterbox), and keep StyleTTS2 and Piper as fallbacks.

---

## 5. What I could NOT verify
- Any V100 latency/RTF/VRAM figure for any model above (none published). The A/B test is the only source.
- Whether SGLang, vLLM ≥ 0.20, or TRT ≤ 10.4 run on sm_70 (only a user forum report for vLLM).
- The Higgs v2 weight license terms (HF says "other, see LICENSE").
- Whether Chatterbox Turbo keeps the exaggeration knob, and its full tag list.
- Whether CosyVoice `inference_instruct2` honours `[laughter]` and `[breath]` together with an instruction.
- Step-Audio-EditX weight license (code Apache; weights unstated on the card).
- Fish S2.1 Pro and Maya 2 Flash open-weight status (on the AA leaderboard without HF links).
- The AA leaderboard page shows no snapshot date; ranks cited were read on 2026-10-08.

## Sources
- IndexTTS: https://github.com/index-tts/index-tts · license https://github.com/index-tts/index-tts/blob/main/LICENSE · https://huggingface.co/mlx-community/IndexTTS-2-MLX · Faster IndexTTS-2 https://arxiv.org/abs/2607.21042
- Chatterbox: https://github.com/resemble-ai/chatterbox · https://huggingface.co/ResembleAI/chatterbox-turbo · https://www.resemble.ai/learn/models/chatterbox
- Fish Audio S2: https://github.com/fishaudio/fish-speech · https://huggingface.co/fishaudio/s2-pro · https://arxiv.org/pdf/2603.08823
- Higgs Audio: https://github.com/boson-ai/higgs-audio · https://github.com/boson-ai/higgs-audio/blob/main/README_V2.md · https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base · https://huggingface.co/bosonai/higgs-audio-v3-tts-4b
- Step-Audio-EditX: https://github.com/stepfun-ai/Step-Audio-EditX · https://huggingface.co/stepfun-ai/Step-Audio-EditX
- Breeze TTS 2: https://huggingface.co/BreezeBlue/Breeze-TTS-2 · https://www.mindstudio.ai/blog/breeze-tts-2-open-weight-model
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice · https://arxiv.org/pdf/2412.10117 · https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- VoxCPM: https://github.com/OpenBMB/VoxCPM
- Orpheus: https://github.com/canopyai/Orpheus-TTS
- Sesame CSM: https://github.com/SesameAILabs/csm
- Dia / Dia2: https://github.com/nari-labs/dia · https://github.com/nari-labs/dia2
- Zonos: https://github.com/Zyphra/Zonos
- Maya1: https://huggingface.co/maya-research/maya1
- VibeVoice: https://github.com/microsoft/VibeVoice
- Voxtral TTS: https://the-decoder.com/mistrals-first-open-weight-tts-model-voxtral-clones-voices-from-three-seconds-of-audio-across-nine-languages/
- F5 / Spark / MaskGCT / MegaTTS3: https://huggingface.co/ScreamingTony/AffectF5 · https://huggingface.co/SparkAudio/Spark-TTS-0.5B · https://huggingface.co/amphion/MaskGCT · https://huggingface.co/ByteDance/MegaTTS3
- GLM-TTS: https://arxiv.org/pdf/2512.14291 · dots.tts: https://arxiv.org/pdf/2606.07080 · EmergentTTS-Eval: https://arxiv.org/pdf/2505.23009
- Rankings/surveys: https://artificialanalysis.ai/text-to-speech/leaderboard · https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/
- V100 tooling: https://newreleases.io/project/pypi/vllm/release/0.10.2 · https://discuss.vllm.ai/t/support-for-v100-sm-70-on-vllm-0-20/2605 · https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.5.0.html
- Vale / ChatGPT voices: https://techcrunch.com/2024/09/24/openai-rolls-out-advanced-voice-mode-with-more-voices-and-a-new-look · https://www.technologyreview.com/2024/09/24/1104422/openai-released-its-advanced-voice-mode-to-more-people-heres-how-to-get-it
