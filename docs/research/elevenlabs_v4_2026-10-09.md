# ElevenLabs Eleven v4: research for an expressive Fish reference (2026-10-09)

**Purpose.** Iris's local voice is Fish Audio S2 Pro, cloned from a reference clip. Fish copies the delivery of its reference, and the current reference is calm all the way through, so she sounds monotone. The plan is for Zeke to render one long, very expressive script in ElevenLabs with the saved Iris voice, then slice that audio into Fish references. This note covers what v4 is, how to direct it, what it costs, and how to write the script.

Labels used below: **[DOC]** = stated in official ElevenLabs docs, blog, changelog, or pricing pages. **[3P]** = third-party or community source. **[INF]** = my own inference.

---

## 1. What Eleven v4 is

- **Model IDs [DOC]:** `eleven_v4` (top quality, aimed at content creation, audiobooks, and character voiceovers) and `eleven_v4_turbo` (real-time, about 100 ms median inference). Source: [Eleven v4 overview](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4), [Models](https://elevenlabs.io/docs/overview/models).
- **Release [DOC]:** **September 28, 2026**, according to the [changelog entry](https://elevenlabs.io/docs/changelog/2026/9/28.md). It is available in the app, through the API, and on the free tier ([landing page](https://elevenlabs.io/v4)).
- **Changes from v3 [DOC]:**
  - New architecture, "the first in a new generation of models."
  - Better quality, voice accuracy, consistency, emotion, and tag-following.
  - 10,000-character cap (v3: 5,000) and 90+ languages (v3: 70+).
  - Professional Voice Clones are supported again.
  - A voice spoken in a language other than its reference's comes out with a native accent, not the reference accent.
  - The docs warn that **a v4 voice "may sound substantially different from Eleven v3,"** because v4 copies the source more faithfully.

## 2. How expressiveness is controlled

**Audio tags [DOC].** Tags are free-form natural-language instructions in **square brackets**. They are "not an enum parameter" ([Text to Dialogue](https://elevenlabs.io/docs/overview/capabilities/text-to-dialogue)). Rules:

- Put a tag before the words it should affect (or just after them, for reactions).
- A tag's emotion **carries forward** until you change it.
- Combine qualities with commas: `[whispering, playful]`.
- Use one tag per clause. Contrasting tags on the same words blur the delivery.
- Parentheses or curly braces may be **read aloud**. Use square brackets only.

([Audio tags list blog](https://elevenlabs.io/blog/elevenlabs-audio-tags-list))

**Published tag list [DOC]** (same blog post; all of these are examples, not a closed set):

| Category | Tags |
|---|---|
| High energy | excited, playful, amazed, powerful, proud, optimistic, startled |
| Heated | mad, aggressive, bitter, critical, repelled |
| Tense | anxious, stressed, scared, threatened, vulnerable, confused, busy |
| Low energy | tired, bored, distant, despair, let down |
| Calm / reflective | peaceful, content, curious, thoughtful, trusting |
| Delivery / volume | whispers, shouts, softly, quietly, low threatening, hushed, barely audible, booming, disbelief |
| Pacing | slowly, rushed, pause, drawn out, snappy, long pause, speedy, speeding up |
| Reactions | laughs, sighs, gasps, clears throat, crying, starts crying |
| Accents / characters | British / French / Australian / Irish accent, pirate voice |
| Sound effects | thunder rumbling, footsteps, door creaking, clapping, dog barking … |

The [prompting guide](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices#prompting-eleven-v4) adds more tags:

- **Laughter and breath:** `[laughs harder]`, `[starts laughing]`, `[wheezing]`, `[exhales]`, `[snorts]`
- **Tone:** `[sarcastic]`, `[mischievously]`
- **Experimental:** `[strong X accent]`, `[sings]`
- **Used by the app's "Enhance" button:** `[chuckles]`, `[short pause]`, `[inhales deeply]`, `[exhales sharply]`, `[appalled]`, `[annoyed]`

**Director-style direction [DOC].** Long descriptive tags work and appear throughout the official examples:

- `[starting calm, then losing patience]`
- `[like a sports commentator, speeding up]`
- `[Low, steady voice, restrained urgency]`
- `[Voice rising into firm resolve]`
- `[Gentle laugh, then sincere]`
- `[Gradually building energy]`

The docs also accept scene-level tags such as `[auctioneer]`. A separate "Director's Mode" is described as *in development*, not shipped ([best practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)). Narrative lines such as *"she said, voice trembling"* also steer the delivery, but **those words get spoken**.

**Voice settings [DOC].** v4 has only **Stability** and **Similarity**.

- Lower Stability gives a more expressive, varied delivery. Higher Stability keeps it close to a fixed baseline.
- Higher Similarity sticks closer to the reference, "at the cost of some naturalness."
- **Style and Speed sliders do not exist in v4. SSML (including `<break>`) is not supported.** Pauses come from tags, ellipses, and text structure.

**Dialogue / multi-speaker [DOC].**

- Use the Text to Dialogue API (`Speaker 1:` / `Speaker 2:` lines, one voice per speaker) or `------------` separators between takes.
- For overlaps, put an em dash where the interruption happens and tag the next line `[jumping in]`, `[overlapping]`, or `[interrupting, then stopping abruptly]` (see the guide's "Overlapping timing" example).

**The UI "get started" cards Zeke sees.** These sit inside the signed-in app, which I did not access. The public site shows a different preset set ("Meet the villain," "Unravel a cold case," "Recall a cherished memory," and others). Each of those is a short script already full of tags such as `[chuckles] … [long pause] … [evil laugh]`. **[INF]** Zeke's cards are probably the same kind of preset, and each one shows a documented feature:

- **Feel the betrayal:** an emotional arc inside one passage (hurt turning to bitterness), using tag progression plus carry-forward.
- **Laugh uncontrollably:** the non-verbal tags (`[starts laughing]`, `[laughs harder]`, `[wheezing]`).
- **Direct a predicament:** director-style compound tags such as `[nervous, trying to sound confident]`.
- **Narrate a mystery:** long-form narration with pacing tags (`[Building tension, measured pace]`).
- **Overlap speech:** the multi-speaker dialogue / interruption feature.
- **Command the room:** volume and power (`[booming]`, `[powerful]`, `[shouts]`).

## 3. Practical limits, cost, and terms

- **Length [DOC]:** 10,000 characters per request, about 10 minutes of audio ([Models](https://elevenlabs.io/docs/overview/models)). The blog says the same cap applies in the Text to Speech app. ⚠ An older help-centre page still lists 5,000 (paid) / 2,500 (free) per website generation and does not mention v4 ([page](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/whats-the-maximum-amount-of-characters-and-text-i-can-generate)). **[INF]** The free tier may still be capped lower, so check the counter in the app.
- **Consistency over long passages [DOC]:** the docs say v4 "preserves speaker identity across long generations." The landing page claims "context stitching" for pacing and "speaker stability" across regenerations. The docs also say v4 "works best when used for paragraphs or passages," not single lines. Independent long-form reports are thin **[3P]**. One aggregator review calls tag adherence "a vendor claim" and advises testing on your own scripts ([OrcaRouter](https://www.orcarouter.ai/blog/eleven-v4-audio-tags-and-instant-voice-cloning)).
- **Saved Voice Design voice [DOC]:** it works, with a caveat. The FAQ says: *"Voice Design voices work with Eleven v4, but they may not be as performative or sound as good as with earlier models."* The landing page adds that voices made before v4 "must be retrained to work effectively"; that retraining step is documented for PVCs. **[INF]** This matters for the plan:
  - A v4 render could sound less like the Iris that Zeke designed, and the docs explicitly warn about v3→v4 drift.
  - The same tags work on `eleven_v3` ([audio tags FAQ](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-do-audio-tags-work-with-eleven-v3-and-v4)), and in-app credit cost is the same.
  - So **render a short tagged test paragraph on both v3 and v4 and let Zeke pick by ear** before spending credits on the long script.
- **Cost [DOC]:**
  - **In the app:** v3 costs **1 credit per character** ([v3 cost FAQ](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-much-does-it-cost-to-generate-using-eleven-v3-alpha)), and the v4 landing page says v4 "uses the same credit pricing as other TTS models." **[INF]** So v4 is also 1 credit/char. Audio tags count toward the total ([character-length FAQ](https://help.elevenlabs.io/hc/en-us/articles/49101802985105-How-is-character-length-calculated)).
  - **Plans:** Free = 10,000 credits/month. Starter = $6/month, about 30,000 credits ([landing](https://elevenlabs.io/v4)).
  - **API:** v4 = $0.08 per 1K characters (launch rate $0.022 until **Oct 12**); v4 Turbo = $0.04 (launch rate $0.011); v3 = $0.08 ([API pricing](https://elevenlabs.io/pricing/api)). There is also a "3× credits on Creator+ until Oct 12" promotion.
  - **[INF]** A 3,000–4,000-character script costs 3–4K credits plus regenerations. That fits inside the free monthly allowance, so **no purchase is needed.**
- **Terms [DOC]:**
  - The free plan has **no commercial license** and requires attribution if content is published ([publishing FAQ](https://elevenlabs.io/docs/help-center/legal/can-i-publish-the-content-i-generate-on-the-platform)). The user keeps rights in the Output ([Terms §4](https://elevenlabs.io/terms)).
  - ⚠ The **[Prohibited Use Policy §12](https://elevenlabs.io/use-policy)** bans *"Using our Services or Output to develop, train, fine-tune, or improve any AI model or service, or including Output in a dataset used for any of those purposes."* **[INF]** Fish zero-shot cloning *conditions* on a reference and does not train weights, so it arguably falls outside that wording. It is close enough that it is **Zeke's call, not mine**, and the risk grows if the clips are ever used to fine-tune a model.
  - ElevenLabs also **watermarks** its audio ([watermark FAQ](https://elevenlabs.io/docs/help-center/legal/audio-detector/what-is-watermarking-and-why-is-eleven-labs-using-it)). **[INF]** The watermark is inaudible, but it would ride along in any Fish reference.

## 4. Writing the expressive script for v4

Sources: [prompting guide](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices#prompting-eleven-v4), [emotional TTS blog](https://elevenlabs.io/blog/emotional-text-to-speech-with-eleven-v4), [tags blog](https://elevenlabs.io/blog/elevenlabs-audio-tags-list). Items marked **[INF]** are specific to the Fish slicing plan.

1. **Write whole passages, not isolated lines.** The model reads emotion from context, and a full scene gives it more to work with [DOC].
2. **Use one emotion per clause, and change tags only where the delivery should change.** Tags carry forward, and stacked contradictory tags blur the result [DOC].
3. **Square brackets only.** Tags should describe the *voice* (`[low, gravelly voice]`), not something that could be read as a sound cue, which can trigger a sound effect. Avoid sound-effect tags entirely in a reference script **[INF]**.
4. **Use punctuation as the pacing control.** Ellipses add pauses and weight. Em dashes cut a line off. Exclamation marks add intensity [DOC]. Use paragraph breaks between beats; there is no SSML.
5. **Use CAPS for a single emphasized word** ("a VERY long day") [DOC]. Keep it sparse **[INF]**.
6. **Use director-style compound tags for arcs:** `[starting calm, then losing patience]`, `[softening, reflective]` [DOC].
7. **Match tags to the voice's character.** A calm designed voice may not do giggles or shouting convincingly. Start with neighbouring emotions, and when a take misses, swap to an adjacent tag instead of rewriting the line [DOC].
8. **Do not put stage directions in plain prose** ("she said sadly"). They get spoken [DOC].
9. **Use low-to-mid Stability for variety. Do not max out Similarity** [DOC trade-off; values **INF**].
10. **Write for the slicer [INF]:**
    - Give each emotion a block of about 15–30 seconds of *speech* (roughly 250–450 characters) in one register.
    - Separate blocks with a paragraph break or `[long pause]` so they can be cut cleanly.
    - Avoid laughs or breaths at block edges, since they make poor clip boundaries.
    - Keep Iris's own words and vocabulary.
    - Generate 2–3 takes of the full script and keep the best block from each [DOC: "generate several and pick"].
