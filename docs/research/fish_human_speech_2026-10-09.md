# Making the Fish S2 Pro mouth sound human: research notes, 2026-10-09

Scope: this note only reports research. It changes no code. The target is `voice/wren_fish_server.py` (Fish S2 Pro on the V100) as launched by `scripts/server/mouth_launch.sh`, which sets `FISH_CHUNK_SCHEDULE=8,16,24` and `FISH_FIRST_FLUSH_S=2.0`, plus the thinking fillers in `voice/wren_voice_core.py` (`CALL_FILLERS`).

Evidence labels used below: **[doc]** means official Fish docs, repo, or paper. **[partner]** means a Fish integration partner's guide. **[community]** means an issue or third-party anecdote. **[research]** means a peer-reviewed study of speech. **[inference]** means my own reasoning, which needs an ear test.

## What the code does today (facts that matter here)

- **Seed:** `FISH_SEED` defaults to `1234` and the launch script does not override it. The voice core sends no seed. So **every utterance is sampled with the same seed**, and identical text such as "Hmm." comes out as an identical waveform each time.
- **Chunking:** Chunks are cut by word budget (8, 16, 24, then ×1.6). The planner prefers punctuation inside the window `[ceil(max/2), max]` and otherwise cuts at a plain word boundary (strength 1).
- **Joins:** Silence at a chunk join is forced to fixed values: 280 ms after a sentence end, 160 ms after a clause, and 60 ms after a plain word. The lead is cut to 10–20 ms. Pauses *inside* a chunk are whatever Fish generated, so the two kinds of pause follow different rules.
- **History:** `FISH_CONTEXT_CHUNKS=1`. History resets with every utterance, so each reply starts "cold" from the reference.
- **Style tags:** A style tag (from the emotion mapping or inline) is carried onto **every** later chunk of the utterance, across sentence boundaries.
- **In-chunk flush:** This already exists. Partial audio is decoded with 24 frames of left context, crossfaded over 1024 samples, and emitted at a deadline (`FIRST_FLUSH_S`, then `audio_end − 0.18 s`). The launch script sets the first deadline to 2.0 s so that it *outlasts* an 8-word chunk, which means the flush is effectively unused for the first chunk.
- **Reference:** `ref_44100.wav` is **13.4 s**, a single clip, with slow, reflective content ("I think slowly, on purpose…").
- **Text cleaning:** The open-source Fish path does no text normalization. The `normalize` field in `fish_speech/utils/schema.py` is never used on the inference path, and the maintainers say the hosted engine "may adopt a different normalization strategy" ([#1268](https://github.com/fishaudio/fish-speech/issues/1268)) **[doc]**. Nothing between `voice_speak` and the mouth strips markdown or symbols.
- **Upstream long-text behavior:** `generate_long` keeps **all** previous batches (text plus codes) in the conversation. It splits only on `<|speaker:N|>` turns and byte budgets of 200–300 bytes, and never into few-word pieces (`fish_speech/models/text2semantic/inference.py`) **[doc]**.

## 1. Top changes, ranked by expected impact vs. effort

### 1. Random seed per utterance (impact: medium, effort: trivial)
- **Change:** Set `FISH_SEED=-1` in `mouth_launch.sh`. Keep one seed per utterance; `_seed_utterance` already does that and logs `used_seed` in `/stats`.
- **Why:** Humans never say "Hmm." the same way twice. Identical tokens in identical contexts are a strong "machine" cue, and with a fixed seed every thinking filler and every repeated opener is bit-identical. Fixing the seed does not make the *voice* consistent; consistency comes from the reference (issue [#1260](https://github.com/fishaudio/fish-speech/issues/1260): seed plus low temperature did not stabilize the voice) **[community]**.
- **Ear test:** Have it say "Hmm." five times and "Okay, so." five times. They should differ slightly and none should sound bad.
- **Risk:** An occasional bad take can no longer be reproduced by default. The logged seed covers that.

### 2. Chunk on prosodic boundaries and let the flush handle latency (impact: high, effort: medium)
See §2 for the details. Change `ChunkPlanner.next` and `_produce_progressive` in `wren_fish_server.py`, and the env in `mouth_launch.sh`.

### 3. Natural pause lengths at chunk joins (impact: medium-high, effort: low)
- **Change:** In `_trim_edges`, `_TAIL_KEEP`, and `_trim_tail`, stop forcing a fixed tail. Keep Fish's own trailing silence, clamped to a band, and add a little jitter:

  | Boundary | Today | Suggested floor / cap |
  |---|---|---|
  | Sentence (`.?!`) | 280 ms | 350–700 ms |
  | Ellipsis (`…`) | 280 ms | 500–900 ms |
  | Clause (`,;:—`) | 160 ms | 150–350 ms |
  | Paragraph / topic shift | none | ~800 ms |
  | Plain word (should become rare, see §2) | 60 ms | 0–80 ms |

  Allow a ±15 % random jitter.
- **Why:** Silent pauses in speech fall into three duration bands: brief (<200 ms), medium (200–1000 ms), and long (>1000 ms, found only in spontaneous speech) ([Campione & Véronis 2002](https://mail.sprosig.org/sp2002/pdf/campione-veronis.pdf)) **[research]**. A constant 280 ms after every sentence sits at the bottom of the medium band and never varies, which is what makes the rhythm sound metronomic. Pauses inside a chunk already vary because Fish generates them. Pauses at joins are currently the only ones that don't.
- **Ear test:** Play a 4–5 sentence reply. It should breathe between sentences without sounding like a list being read.
- **Risk:** Total utterance time grows by about 0.1–0.3 s per sentence. Barge-in is unaffected.

### 4. A longer, conversational reference (impact: high, effort: medium)
- **Change:** Replace or extend `~/voice_lab/ref/ref_44100.wav` and `ref.txt` with **20–30 s** of Iris's voice speaking *conversationally*. It should include a question, a contraction-heavy casual sentence, one natural "um", a light laugh, and both a warm and a matter-of-fact sentence, with about 0.5 s between sentences. The transcript must match **exactly**, including the "um" and punctuation. Two clips are possible: upstream concatenates multiple prompt clips, and for a single speaker both should carry `<|speaker:0|>`.
- **Why:**
  - Fish says cloning "captures timbre, speaking style, and emotional tendencies" ([README](https://github.com/fishaudio/fish-speech)) **[doc]**.
  - The recommendations are 10–30 s ([HF card / README](https://huggingface.co/fishaudio/s2-pro)), or 2–3 clips of 15–20 s with half-second pauses ([voice-cloning best practices](https://docs.fish.audio/developer-guide/best-practices/voice-cloning.md)) **[doc]**.
  - Their fix for robotic output is "record for longer, 30–60 seconds" and "speak more naturally and add pauses" (same page, and the [Comfy partner guide](https://docs.comfy.org/tutorials/partner-nodes/fishaudio/prompt-guide.md)) **[doc]/[partner]**.
  - The SDK guide asks for "varied intonation and emotion" and transcripts that match exactly, punctuation included **[doc]**.
  - The current 13.4 s reference is short, a single clip, and slow and contemplative in delivery. That delivery is likely being imposed on everything **[inference]**.
- **Ear test:** Compare the same five test replies (casual, excited, comforting, technical, question) under the old and new reference.
- **Risk:** A 30 s reference is about 650 frames in the KV cache, versus about 290 now. That leaves less headroom under `FISH_MAXSEQ=4096` for history plus generation, but it still fits. Decode attends over a longer prefix, so re-measure RTF in `/stats`. The reference is cached, so there is no per-chunk prefill cost.

### 5. A spoken-style text layer (impact: high, effort: low)
- **Change (a): how Iris writes for the ear.** Add this as a rule in the voice prompt or memory:
  - Use contractions.
  - Keep sentences to about 8–20 words and vary their length.
  - Start with discourse markers ("so", "okay", "well", "honestly") where a person would.
  - No lists, markdown, parentheticals, emoji, or URLs. Write numbers, units, and symbols as words.
  - Always use end punctuation, with "?" on real questions.
  - Keep spoken replies to 1–3 sentences unless asked for more.

  This matches [Inworld's TTS prompting guide](https://docs.inworld.ai/tts/best-practices/prompting-for-tts) ("Use contractions…", "Never use markdown…", "Write everything as natural spoken sentences", "vary sentence length") and [ElevenLabs agents](https://elevenlabs.io/docs/agents-platform/best-practices/prompting-guide) ("under 3 sentences") **[partner, other vendors]**.
- **Change (b): a deterministic normalizer.** Put it in the Fish server ahead of `_units()`. It should strip `* # _ \` ~`, bullets, and emoji; expand `&`, `%`, `$`, `/`, and `@`; drop URLs; and leave `[tags]` intact. Rationale: the open-source path does none of this (#1268), and digits and symbols are the most common mispronunciation source ([ElevenLabs](https://elevenlabs.io/docs/agents-platform/best-practices/prompting-guide)).
- **Ear test:** Try a reply containing "~3 GB, 50% faster (see below)".
- **Risk:** Over-normalizing proper nouns. Keep the rules few.

### 6. Less tag-carrying, more meaningful tags (impact: medium, effort: low)
- **Change:** In `ChunkPlanner.next`, reset `self.style` at sentence boundaries (strength 3) instead of carrying it through the whole reply. In `_style_from_obj`, apply the param-mapped tag only for intensity ≥ 0.6, and only to the first sentence. Prefer inline tags that Iris writes where the text supports them (`[chuckling] heh`, `[sigh]`, `[soft voice]`, `[emphasis]`).
- **Why:**
  - Fish: a tag applies "until the next tag or the end of the sentence". Use "one primary emotion per sentence", "≤3 tags per sentence", "don't overuse tags in short text", and sentence-level cues work best at the sentence start ([emotion reference](https://docs.fish.audio/api-reference/emotion-reference)) **[doc]**. Today's carrying extends a tag beyond Fish's own scope rule.
  - Tags on emotion-neutral text are "very subtle to essentially inaudible" ([#1280](https://github.com/fishaudio/fish-speech/issues/1280), confirmed by two other users) **[community]**. So `[happy]` on a neutral sentence is mostly noise. Matching the words to the feeling matters more than the tag.
  - The S2 paper's Audio Turing Test score rose from 0.483 to 0.515 when an LLM rewrote the text with inline instructions ([arXiv 2603.08823](https://arxiv.org/html/2603.08823v2)) **[doc]**. That means LLM-placed tags in the right places help.
  - Laugh and sigh tags should be followed by matching text ("Ha, ha", "heh") ([Comfy guide](https://docs.comfy.org/tutorials/partner-nodes/fishaudio/prompt-guide.md)) **[partner]**.
- **Ear test:** Play a long joyful reply with and without carrying. The carried version should sound more like a performance.
- **Risk:** Emotion becomes less "on" overall. That is the intent.

### 7. More context: 2 chunks, and carry across close utterances (impact: medium, effort: low-medium)
- **Change:** Set `FISH_CONTEXT_CHUNKS=2`. Keep the last chunk of the previous utterance as history for the next one if it is under about 60 s old and was not barge-in-stopped, by moving `history` out of `_produce_progressive` into a module-level slot.
- **Why:** Upstream keeps the *whole* prior conversation for long text, and S2 was trained for multi-turn generation that "leverage[s] previous information to improve the expressiveness of subsequent generated content" ([README](https://github.com/fishaudio/fish-speech)) **[doc]**. A community session-state patch reports 3–4× fewer seams than stateless chunking ([discussion #1300](https://github.com/fishaudio/fish-speech/discussions/1300)) **[community]**.
- **Ear test:** A back-and-forth of three replies. The energy level should flow from one to the next instead of resetting each time.
- **Risk:** The prefill grows by the history tokens on every chunk, so measure `prefill_s`. Emotion can bleed between replies, which is the reason for the 60 s cutoff. It also uses KV headroom.

### 8. Sampling A/B (impact: low-medium, effort: trivial)
See §4.

### 9. int8 vs. fp16 expressiveness A/B (impact: unknown, possibly medium; effort: trivial)
- **Change:** Run `FISH_INT8=0` for a listening session.
- **Why:** The server's own comment flags int8 as an ear-check item. Quantization error in the slow AR could plausibly flatten prosody **[inference]**. fp16 measured RTF ~0.65, which is still real-time.
- **Risk:** Higher VRAM (the file states ~9–12 GB with int8) and less headroom for longer chunks.

### 10. Breaths, and protecting them from the trimmer (impact: low-medium, effort: low)
- **Change:** Allow `[inhale]` before long (>20-word) sentences that follow a sentence pause, at most once per reply. Check that `_trim_lead` (RMS threshold 0.004, 10–20 ms kept) isn't cutting generated inhales; try keeping 150 ms of lead after a sentence boundary.
- **Why:**
  - Breaths in spontaneous speech mostly fall at boundaries. In read speech 96.8–100 % fall at syntactic boundaries; in spontaneous speech 13–31 % fall *inside* constituents ([Włodarczak & Heldner 2017](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00708/full)) **[research]**.
  - Synthesized breathing helps naturalness and recall, but copying *disfluent* breathing hurts ([Székely et al., ICASSP 2020](https://www.speech.kth.se/tts-demos/szekely2020breathing.pdf)) **[research]**.
  - Fish's tags include `[inhale]`, `[exhale]`, `[breathing]`, and `[sigh]` ([Fish blog](https://fish.audio/blog/how-to-use-inline-tags-in-fish-audio-s2/)) **[doc]**.
- **Risk:** Too many breaths sound asthmatic. Keep it to one per reply at most.

### 11. Speaker token on the target text (impact: unknown, effort: trivial)
- **Change:** A/B prefixing `<|speaker:0|>` to each chunk's user text in `_encode`. The reference text already carries it, and the upstream CLI default text uses it. The upstream API does not add it.
- **Why:** This might bind the target more tightly to the reference speaker. It is **speculative [inference]**, so test before adopting.

### 12. Text cleanup of the filler set (impact: medium, effort: trivial)
See §3.

## 2. Chunk boundaries: fixed 8 words vs. prosody-aware

**The arithmetic.** At ~26 ms of compute per 46.4 ms frame and ~7 frames per word, one word is ~325 ms of audio (~185 wpm) and ~180 ms of compute. A fixed 8-word first chunk is therefore ~1.45 s of steps plus prefill, about 1.8–2 s to first audio, which matches what was measured.

**Why 3-word chunks sounded choppy [inference, consistent with the literature].** Each chunk is a separate generation, so Fish renders it as a *complete phrase*: a pitch reset at the start, declination, and a final slowdown and pitch fall at the end.
- Speech tempo depends mainly on phrase length, and short phrases are slower because of the final ritard ([Quené 2005](https://www.isca-archive.org/interspeech_2005/quene05_interspeech.html); [Yuan & Cieri, Language Log](https://languagelog.ldc.upenn.edu/nll/?p=1322)) **[research]**.
- So every cut that lands mid-phrase manufactures a false phrase ending. Eight words reduces the number of cuts but does not prevent false endings, because the planner still cuts at plain words when no punctuation falls in the window.
- Fish's own guidance points the same way: "Use punctuation for natural pauses", "Buffer 5–10 words", don't send characters ([realtime streaming](https://docs.fish.audio/developer-guide/best-practices/real-time-streaming)) **[doc]**. Deepgram: sentence boundaries are "the simplest and most effective approach", and larger chunks preserve intonation ([Deepgram](https://developers.deepgram.com/docs/tts-text-chunking)). LiveKit's "expressive" mode batches whole sentences to keep prosody continuous.

**Key observation: latency doesn't have to come from chunk size here.** The full reply text is already known when `speak_progressive` starts, and the server already has an in-chunk flush. With RTF ~0.55, generation runs about 1.8× faster than playback. So once a chunk has started, deadline flushes can never underrun. Each flush emits more audio than has played since the previous one.

Set the first flush deadline to ~1.0 s. Assuming ~0.3 s of prefill (the true figure is in `prefill_s`), about 27 frames, ≈1.25 s of audio, are ready at 1.0 s. That gives **~1 s to first audio whatever the length of the first chunk**. That is better than today's ~2 s, and the first chunk can still be a whole clause.

**Recommendation:**
1. **Chunk unit = sentence.** Merge sentences shorter than 6 words with the next one ("Yeah. I think so." becomes one chunk). Cap at ~28 words; above the cap, split at the strongest clause boundary (`; : —`, then `,` before a conjunction) nearest the middle. Cut at a plain word (strength ≤1) only if no punctuation falls within 28 words.
2. **First chunk:** the same rule with a ~16-word cap, preferring the first clause boundary at 4 words or more.
3. **Latency:** In `mouth_launch.sh`, set `FISH_FIRST_FLUSH_S≈1.0`, keep `FISH_FLUSH=1` and `FLUSH_LEAD_S=0.18`, and drop `FISH_CHUNK_SCHEDULE` to a guard role (e.g. `16,28,28`). The live speed model then only matters if flush is off.
4. **Verify in dry-run** (`IRIS_MOUTH_DRYRUN=1`): check underrun gaps, `codec_s` per flush (every flush is an extra codec decode with 24 frames of context on the same GPU thread), and listen at the flush seams. Seam quality is the open question. If seams are audible, fall back to **first chunk = first clause (4–10 words, cut at a comma) with no flush, then whole sentences**. That keeps ~1–1.6 s first audio and still puts every cut on a real boundary.

## 3. Filler policy

**Does Fish render "uh/um/hmm" naturally?** Mostly yes:
- Fish's partner guide: "Plain pause words also shape rhythm without any tags: 'um', 'uh'" ([Comfy](https://docs.comfy.org/tutorials/partner-nodes/fishaudio/prompt-guide.md)) **[partner]**.
- S2's training transcripts were annotated "alongside natural disfluencies" ([paper](https://arxiv.org/html/2603.08823v2)) **[doc]**.
- Zeke's own ear test on 10-09 found that hums come out as hums.
- **No filler tag is documented.** The free-form `[hesitant]` / `[speaking slowly, almost hesitant]` ([Fish blog](https://fish.audio/blog/how-to-use-inline-tags-in-fish-audio-s2/)) is an option to try, not a requirement.

**What human data says:**
- "uh" signals a short delay and "um" a longer one ([Clark & Fox Tree 2002](https://gwern.net/doc/psychology/linguistics/2002-clark-2.pdf)). In Switchboard, "um" is followed by silence 74 % of the time vs. 49 % for "uh". Median durations are 417 ms for "um" and 286 ms for "uh". "uh" is about 5× more frequent overall (67k vs. 12.5k in 2.7M words, ≈2.5–3 fillers per 100 words) ([Language Log](https://languagelog.ldc.upenn.edu/nll/?p=14991)) **[research]**.
- Fillers cluster at the start of utterances and **just before complex constituents** ([CUNY 2019 review](https://www.colorado.edu/event/cuny2019/sites/default/files/attached-files/a30_rose.pdf)) and track planning difficulty ([Bortfeld et al. 2001](https://researchconnect.stonybrook.edu/en/publications/disfluency-rates-in-conversation-effects-of-age-relationship-topi/)) **[research]**.
- Fillers make a synthetic speaker sound more uncertain and casual ([Gustafson et al., SSW 2021](https://www.isca-archive.org/ssw_2021/gustafson21_ssw.html)). Listeners preferred fillers placed where real speakers put them ([Székely et al., SSW 2019](https://people.kth.se/~ghe/pubs/pdf/szekely2019how.pdf)) **[research]**.
- Turn gaps have a mode of 0–200 ms across 10 languages ([Stivers et al. 2009](https://forms.mpi.nl/node/50939)). Gaps of about 700 ms or more start to read as a coming "no" or trouble ([Kendrick & Torreira 2015](https://forms.mpi.nl/node/50430)) **[research]**.

**Policy:**
1. **Thinking fillers (`CALL_FILLERS`):**
   - Spellings: use standard forms: "Hmm.", "Hmm…", "Um…", "Uh…", "Mm.", "Okay, um…", "Let's see…". Drop "Uhh..." and "Uhm," because non-standard spellings tokenize unpredictably (ear-test any you keep).
   - Choice: pick **randomly**, not round-robin, with no repeat among the last 3. A fixed seed would undo this (§1.1).
   - Timing: fire ~300–600 ms after his turn ends whenever real audio won't start within ~1 s. Use "Hmm"/"um" (long-delay signals) because the cognition gap is seconds, not milliseconds.
   - Limit: one filler per turn. A second, verbal bridge ("give me a sec…") only after ~4–5 s, and only once the echo problem behind `STALL_BRIDGE=False` is solved.
2. **Fillers inside replies (written by Iris):**
   - Rate: at most **one per 2–3 sentences**, well below the human rate of ~1 per 35–40 words. A misplaced synthetic filler costs more than a missing one.
   - Placement: only at a sentence or clause start, or after a discourse marker ("so, um,"). Only right before genuinely hard or uncertain content: a word search, a number Iris is unsure of, a delicate thing. Use "um" when a pause follows, "uh" for a quick stumble.
   - Never: two in one sentence; inside a fact Iris is sure of (fillers signal uncertainty, so a filler there would misrepresent her confidence); immediately after a thinking filler.
3. **Laughs and sighs** follow the same restraint: pair the tag with text (`[chuckling] heh,`). Provine's speech-laughter work suggests laughs land at phrase ends ([Provine](https://www.cambridge.org/core/product/F15F98C36090A1659ED27B7338A14B9C)). That claim is contested for interviews, so place them at clause ends rather than mid-phrase.

## 4. Sampling parameters to A/B

Current settings: T 0.8, top_p 0.8, top_k 30, with repetition-aware sampling built into the decode step (window 10, fallback T 1.0 / top_p 0.9). `repetition_penalty` isn't used on this path.

Reference points:
- Upstream CLI: T 1.0, top_p 0.9.
- API schema: T 0.8, top_p 0.8.
- Hosted docs and the model-manager warm-up: 0.7 / 0.7, with "The defaults are well tuned" ([Comfy](https://docs.comfy.org/tutorials/partner-nodes/fishaudio/prompt-guide.md)) **[doc/partner]**.
- Third-party guide: 0.7–0.8 is the sweet spot; <0.5 sounds flat; >0.9 risks glitches and timbre drift; raise top_p to 0.9–0.95 if speech slurs ([cosmo-edge](https://cosmo-edge.com/fish-audio-s2-pro-voice-reference-guide/)) **[community]**.

| Run | T | top_p | top_k | Listen for |
|---|---|---|---|---|
| A (baseline) | 0.8 | 0.8 | 30 | — |
| B | 0.7 | 0.7 | 30 | steadier timbre, flatter? |
| C | 0.9 | 0.85 | 30 | livelier pitch, any glitches/timbre drift |
| D | 0.8 | 0.9 | 30 | less slurring, more variety |

Use the same 6 test sentences per run, with random seeds and 3 takes each. Judge on liveliness, glitches, and timbre stability. Leave `top_k` alone: it is baked into the compiled step, so changing it risks a long recompile.

Changing T or top_p needs only an env change plus a restart (~3–4 min with a warm inductor cache). Do it during a planned window, since this is the live mouth.

## Biggest open questions (need ears, not reading)
- Are flush seams audible? This decides which variant of §2 to use.
- How much does a conversational reference shift the default delivery? I expect this to be the biggest single lever.
- Does int8 cost expressiveness?
