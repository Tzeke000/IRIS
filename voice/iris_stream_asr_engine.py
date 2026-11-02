"""iris_stream_asr_engine.py - live cache-aware streaming RNN-T (Nemotron) for the ears sidecar.

Runs ONLY inside the NeMo venv (imported by voice/iris_stream_asr_server.py, never by the ears
daemon). One NemoStreamer holds the model; one StreamSession = one utterance.

Why this file exists (bench note 2026-11-02, "the one unbenchmarked piece"): the offline bench fed
CacheAwareStreamingAudioBuffer a WHOLE clip, so the log-mel was computed once over all of it. Live
audio arrives 100 ms at a time, so here the mel is computed incrementally:

  * a mel frame k is centred on sample k*160 and its STFT window spans samples [k*160-256, k*160+256)
    (n_fft 512, hop 160; pre-emphasis looks back one more sample, normalize = NA, no dither in eval).
    So frame k is FINAL once samples up to k*160+256 exist, and computing it from any audio slice that
    starts >= 3 frames earlier gives the same numbers as the whole-clip pass. Each push recomputes only
    a short slice (new frames + 3 frames of margin) - bounded cost, identical features.
  * at flush the tail frames are computed from a slice that ends at the true end of the audio, so the
    end padding matches the whole-clip pass too.
  * the chunk / pre-encode-cache / drop_extra_pre_encoded schedule copies CacheAwareStreamingAudioBuffer
    exactly (first chunk chunk_size[0] with no cache, then chunk_size[1] with pre_encode_cache_size[1]
    frames of cache). A live step runs only while at least one more final frame exists after the chunk,
    so it can never be the step the offline pass would have run with keep_all_outputs=True.

tools/stream_asr_replay.py replays clips through this path to compare with the bench numbers.
"""
from __future__ import annotations

import copy
import math
import threading
import time

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

SR = 16000
HOP = 160            # 10 ms mel hop
HALF_WIN = 256       # n_fft // 2: samples each side of a frame centre the STFT can touch
MARGIN_FRAMES = 3    # 3 * 160 = 480 > 256 + 1 (pre-emphasis): slice start never touches a kept frame


def _texts(t):
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    t = t[0] if isinstance(t, (list, tuple)) and t and isinstance(t[0], (list, tuple)) else t
    return [h.text if isinstance(h, Hypothesis) else h for h in t]


class NemoStreamer:
    """The loaded model + decoding config. Not thread-safe: the server runs every step on ONE thread."""

    def __init__(self, model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b", right_ctx: int = 1,
                 device: str = "cuda", boost_phrases=None, boost_alpha: float = 0.0,
                 boost_score: float = 1.0):
        import nemo.collections.asr as nemo_asr
        self.model_id = model_id
        self.right_ctx = int(right_ctx)
        self.device = device
        m = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location=device)
        if hasattr(m.encoder, "set_default_att_context_size"):
            m.encoder.set_default_att_context_size([70, self.right_ctx])
        dc = m.cfg.decoding
        with open_dict(dc):
            dc.strategy = "greedy_batch"
            dc.fused_batch_size = -1
            dc.greedy.max_symbols = 10
            # NeMo 3.0 Hypothesis.merge_ trips over dict timestamps across streaming steps (bench note);
            # we time tokens ourselves from the step they appear in (see StreamSession.words).
            dc.compute_timestamps = False
            dc.preserve_alignments = False
            if "loop_labels" in dc.greedy:
                dc.greedy.loop_labels = True
            self.boost_phrases = [p for p in (boost_phrases or []) if p and p.strip()]
            if self.boost_phrases and boost_alpha > 0:
                dc.greedy.boosting_tree = {"key_phrases_list": list(self.boost_phrases),
                                           "context_score": float(boost_score),
                                           "use_triton": False}
                dc.greedy.boosting_tree_alpha = float(boost_alpha)
        m.change_decoding_strategy(dc)
        self.model = m.to(device, dtype=torch.float32).eval()
        self.scfg = self.model.encoder.streaming_cfg
        # feature extractor: same as CacheAwareStreamingAudioBuffer.extract_preprocessor
        pcfg = copy.deepcopy(self.model._cfg.preprocessor)
        OmegaConf.set_struct(pcfg, False)
        pcfg.dither = 0.0
        pcfg.pad_to = 0
        assert str(pcfg.get("normalize", "NA")) in ("NA", "None", "none"), "per-utterance norm can't stream"
        assert int(pcfg.n_fft) // 2 == HALF_WIN and int(round(pcfg.window_stride * SR)) == HOP
        self.pre = self.model.from_config_dict(pcfg).to(device).eval()
        self.n_feat = self.model.encoder._feat_in
        sf = getattr(self.model.encoder.pre_encode, "get_sampling_frames", None)
        self.sampling_frames = sf() if sf else None
        # encoder output frame = subsampling_factor mel frames (80 ms here)
        self.enc_frame_s = int(self.model.cfg.encoder.subsampling_factor) * HOP / SR
        self.chunk_ms = (self.right_ctx + 1) * 80

    @staticmethod
    def _pick(v, first: bool):
        return (v[0] if first else v[1]) if isinstance(v, list) else v

    def session(self) -> "StreamSession":
        return StreamSession(self)

    def warm(self, seconds: float = 2.0, times: int = 2) -> float:
        """Run synthetic utterances end to end (kernels, cuDNN autotune). Returns ms of the last."""
        rng = np.random.default_rng(0)
        dt = 0.0
        for _ in range(times):
            t = time.perf_counter()
            s = self.session()
            a = (0.01 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
            for i in range(0, len(a), 1600):
                s.push(a[i:i + 1600])
            s.finish()
            dt = (time.perf_counter() - t) * 1000
        return dt


class StreamSession:
    """One utterance. push() float32 16 kHz audio as it arrives; finish() once at the end."""

    def __init__(self, st: NemoStreamer):
        self.st = st
        self.audio = np.zeros(0, np.float32)
        self.mel = torch.zeros((1, st.n_feat, 0), device=st.device)   # FINAL frames only
        self.idx = 0                    # next mel frame to feed the encoder
        self.step = 0
        self.cc, self.ct, self.ccl = st.model.encoder.get_initial_cache_state(batch_size=1)
        self.prev = None
        self.pred = None
        self.text = ""
        self.finished = False
        self.step_ms: list = []
        self.tok_times: list = []       # (token_id, enc-frame time s) for tokens in emission order
        self._ntok = 0

    # -- features ---------------------------------------------------------------------------
    def _extend_mel(self, final: bool) -> None:
        n = len(self.audio)
        k0 = self.mel.shape[-1]
        if final:
            end = n
        else:
            if n < HALF_WIN:
                return
            k_end = (n - HALF_WIN) // HOP + 1           # frames whose window is fully inside the audio
            if k_end <= k0:
                return
            end = (k_end - 1) * HOP + HALF_WIN
        s0 = max(0, (k0 - MARGIN_FRAMES) * HOP)
        seg = self.audio[s0:end]
        if seg.size == 0:
            return
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0).to(self.st.device)
            feats, lens = self.st.pre(input_signal=x, length=torch.tensor([seg.size], device=self.st.device))
        g0 = s0 // HOP                                   # global index of local frame 0
        # final: take EVERY frame the preprocessor returns, incl. the one masked tail frame past
        # `lens` - CacheAwareStreamingAudioBuffer sizes its buffer by the tensor, not by lens, and
        # without that frame the last chunk drops words ("...running back [here]", a final "?").
        k_end = g0 + feats.shape[-1] if final else k_end
        j0, j1 = k0 - g0, k_end - g0
        if j1 > j0:
            self.mel = torch.cat([self.mel, feats[:, :, j0:j1]], dim=-1)

    # -- one encoder/decoder step (copied schedule of CacheAwareStreamingAudioBuffer.__iter__) --
    def _run_step(self, final_pass: bool) -> bool:
        st, cfg = self.st, self.st.scfg
        total = self.mel.shape[-1]
        first = self.idx == 0
        cs = st._pick(cfg.chunk_size, first)
        shift = st._pick(cfg.shift_size, first)
        if not final_pass and self.idx + cs >= total:
            return False                                 # need one final frame past the chunk
        if self.idx >= total:
            return False
        chunk = self.mel[:, :, self.idx:self.idx + cs]
        if st.sampling_frames is not None:
            sfr = st._pick(st.sampling_frames, first) if isinstance(st.sampling_frames, list) else st.sampling_frames
            if chunk.shape[-1] < sfr:
                return False
        pcs = st._pick(cfg.pre_encode_cache_size, first)
        if first:
            cache = torch.zeros((1, st.n_feat, pcs), device=chunk.device, dtype=chunk.dtype)
            zeros = None
        else:
            start = max(0, self.idx - pcs)
            cache = self.mel[:, :, start:self.idx]
            zeros = (torch.zeros((1, chunk.shape[-2], pcs - cache.shape[-1]), device=chunk.device,
                                 dtype=chunk.dtype) if cache.shape[-1] < pcs else None)
        added = cache.shape[-1]
        x = torch.cat((cache, chunk), dim=-1)
        if zeros is not None:
            x = torch.cat((zeros, x), dim=-1)
            added += zeros.shape[-1]
        lens = torch.clamp(torch.tensor([total - self.idx + added], device=x.device), min=0, max=x.shape[-1])
        self.idx += shift
        keep_all = final_pass and self.idx >= total
        t = time.perf_counter()
        with torch.inference_mode():
            self.pred, tx, self.cc, self.ct, self.ccl, self.prev = st.model.conformer_stream_step(
                processed_signal=x, processed_signal_length=lens,
                cache_last_channel=self.cc, cache_last_time=self.ct, cache_last_channel_len=self.ccl,
                keep_all_outputs=keep_all, previous_hypotheses=self.prev, previous_pred_out=self.pred,
                drop_extra_pre_encoded=(0 if self.step == 0 else cfg.drop_extra_pre_encoded),
                return_transcription=True)
        self.step_ms.append((time.perf_counter() - t) * 1000)
        self.step += 1
        self.text = ((_texts(tx)[0] or "") if tx is not None else self.text).strip()
        self._note_tokens()
        return True

    def _note_tokens(self) -> None:
        """Time every token that appeared in this step: the encoder frames this step consumed.
        Coarse (one step = chunk_ms), but enough to order words and find long pauses."""
        try:
            h = self.prev[0]
            ys = h.y_sequence
            ys = ys.tolist() if hasattr(ys, "tolist") else list(ys)
        except Exception:
            return
        new = ys[self._ntok:]
        if new:
            t_end = min(self.idx, self.mel.shape[-1]) * HOP / SR     # mel consumed so far, seconds
            t_beg = max(0.0, t_end - self.st.chunk_ms / 1000.0)
            for i, tok in enumerate(new):
                self.tok_times.append((int(tok), t_beg + (t_end - t_beg) * (i + 0.5) / len(new)))
        self._ntok = len(ys)

    # -- public ------------------------------------------------------------------------------
    def push(self, audio_f32) -> bool:
        """Append audio; run every step that is now possible. True if the partial text changed."""
        if self.finished:
            return False
        a = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if a.size:
            self.audio = np.concatenate([self.audio, a])
        before = self.text
        self._extend_mel(final=False)
        while self._run_step(final_pass=False):
            pass
        return self.text != before

    def finish(self) -> str:
        """Flush: final mel frames, then the remaining (possibly partial) chunks; last keeps all outputs."""
        if self.finished:
            return self.text
        self._extend_mel(final=True)
        while self._run_step(final_pass=True):
            pass
        self.finished = True
        return self.text

    def words(self) -> list:
        """[(word, t_start, t_end)] from the step-timed tokens (sentencepiece '▁' starts a word)."""
        tok = self.st.model.tokenizer
        out, cur, t0, t1 = [], "", None, None
        for tid, t in self.tok_times:
            try:
                piece = tok.ids_to_tokens([tid])[0]
            except Exception:
                continue
            if piece.startswith("▁") and cur:
                out.append((cur, t0, t1))
                cur, t0 = "", None
            cur += piece.replace("▁", "")
            t0 = t if t0 is None else t0
            t1 = t
        if cur:
            out.append((cur, t0, t1))
        return out
