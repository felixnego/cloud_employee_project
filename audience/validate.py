"""Checks on the generated panel.

Two kinds. **Integrity** checks are structural and must hold -- broken keys or
emotion rows present without a detected face mean the generator is wrong.
**Recovery** checks are the reason to publish ground truth: fit a plain OLS to
the generated data and confirm it returns the coefficients that produced it. If
it does not, either the generator or the analysis is at fault, and either way
somebody should know before drawing a conclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TOLERANCE = 0.25        # absolute, on logit-scale coefficients


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def integrity(t: dict[str, pd.DataFrame], creative_ids: set[str]) -> list[dict]:
    bio, sess, viewers = t["biometric_seconds"], t["sessions"], t["viewers"]
    emo_cols = [f"p_{e}" for e in config.EMOTIONS]
    present = bio[bio.face_detected]
    missing = bio[~bio.face_detected]
    sums = present[emo_cols].sum(axis=1)

    unknown = set(sess.creative_id) - creative_ids
    over = (sess.watch_time_s > sess.creative_duration_s).sum()
    no_consent = set(viewers.loc[~viewers.consent_biometric, "viewer_id"])

    return [
        _check("sessions reference real creatives", not unknown,
               f"{len(unknown)} unknown creative_id(s)"),
        _check("biometric rows reference real sessions",
               set(bio.session_id).issubset(set(sess.session_id)),
               f"{len(set(bio.session_id) - set(sess.session_id))} orphan session(s)"),
        _check("watch time never exceeds the creative", over == 0,
               f"{over} session(s) longer than the ad"),
        _check("no biometrics without consent",
               not (set(bio.viewer_id) & no_consent),
               f"{len(set(bio.viewer_id) & no_consent)} viewer(s) without consent have rows"),
        _check("emotions NULL exactly when no face detected",
               missing[emo_cols].notna().to_numpy().sum() == 0
               and present[emo_cols].isna().to_numpy().sum() == 0,
               f"{int(missing[emo_cols].notna().to_numpy().sum())} unexpected values"),
        _check("emotion probabilities sum to 1",
               bool(np.allclose(sums, 1.0, atol=1e-3)),
               f"max deviation {float((sums - 1).abs().max()):.5f}"),
    ]


def recovery(t: dict[str, pd.DataFrame], cseconds: pd.DataFrame) -> list[dict]:
    """Refit the attention model and compare against the planted coefficients."""
    bio = t["biometric_seconds"].merge(
        cseconds[["creative_id", "second", "position_in_ad", "n_cuts", "has_face",
                  "has_speech", "has_screen_text", "motion_z"]],
        on=["creative_id", "second"], how="left")

    terms = ["position_in_ad", "cut_density", "has_face", "is_speech",
             "has_screen_text", "motion_z"]
    X = np.column_stack([
        np.ones(len(bio)),
        bio.position_in_ad.to_numpy(),
        bio.n_cuts.to_numpy(),
        bio.has_face.to_numpy().astype(float),
        bio.has_speech.to_numpy().astype(float),
        bio.has_screen_text.to_numpy().astype(float),
        bio.motion_z.to_numpy(),
    ])
    a = np.clip(bio.attention_index.to_numpy(), 1e-6, 1 - 1e-6)
    y = np.log(a / (1 - a))                      # attention is sigmoid(linear)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    out = []
    for i, term in enumerate(terms, start=1):
        truth = config.ATTENTION[term]
        est = float(beta[i])
        out.append(_check(f"recovers beta[{term}]", abs(est - truth) <= TOLERANCE,
                          f"true {truth:+.2f}  estimated {est:+.2f}  "
                          f"delta {est - truth:+.2f}"))

    # The IAT: trials should measure the latent trait they were generated from.
    merged = t["iat_scores"].merge(t["viewers"][["viewer_id", "latent_implicit_pref"]],
                                   on="viewer_id")
    r = float(np.corrcoef(merged.d_score, merged.latent_implicit_pref)[0, 1])
    faster = float(t["iat_scores"].mean_rt_congruent_ms.mean()
                   - t["iat_scores"].mean_rt_incongruent_ms.mean())
    out += [
        _check("IAT D-score tracks the latent trait", r > 0.5,
               f"corr(d_score, latent) = {r:.3f}"),
        _check("congruent trials are faster", faster < 0,
               f"congruent - incongruent = {faster:.1f} ms"),
    ]
    return out


def run_all(tables: dict[str, pd.DataFrame], cseconds: pd.DataFrame,
            creative_ids: set[str]) -> pd.DataFrame:
    return pd.DataFrame(integrity(tables, creative_ids) + recovery(tables, cseconds))
