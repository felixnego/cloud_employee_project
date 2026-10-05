"""Generates a synthetic audience panel, driven by the real creative features.

The point is that the response is *caused by* what was measured in the videos.
Attention at second t is a function of the cuts, faces, speech and motion that
`main.creative_seconds` records for that second, plus who the viewer is, plus
an AR(1) term so traces are smooth rather than independent draws. Random
numbers unconnected to the creatives would exercise nothing and teach nobody.

Causal order, which is also the order of this file:

    viewer traits  ->  IAT trials  ->  D-score        (a noisy measurement of a trait)
    viewer traits  +  creative features  ->  biometrics
    biometrics  ->  survey answers

Everything is drawn from one seeded generator, so a run is reproducible.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd

from . import config

RNG_SECTIONS = ("viewers", "iat", "sessions", "survey")


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    """Autocorrelated noise: a viewer's attention drifts, it does not jump."""
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0, sd * (1 - rho ** 2) ** 0.5)
    return out


# --------------------------------------------------------------- creative --
def load_creative_features(db_path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the measured creative layer. This is the generator's only input."""
    with duckdb.connect(str(db_path or config.DUCKDB_PATH), read_only=True) as con:
        creatives = con.execute("""
            SELECT creative_id, display_name, duration_s, n_shots
            FROM creatives ORDER BY creative_id
        """).df()
        seconds = con.execute("""
            SELECT creative_id, second, position_in_ad, n_cuts,
                   coalesce(has_face, false)                     AS has_face,
                   coalesce(has_speech, false)                   AS has_speech,
                   coalesce(has_screen_text, false)              AS has_screen_text,
                   coalesce(motion, 0)                           AS motion,
                   coalesce(mean_p_happiness, 0)                 AS creative_happiness,
                   (audio_class = 'music')                       AS is_music
            FROM creative_seconds ORDER BY creative_id, second
        """).df()
    # Standardise motion within a creative so a dark, still film and a bright,
    # busy one contribute comparable signal.
    seconds["motion_z"] = seconds.groupby("creative_id").motion.transform(
        lambda s: (s - s.mean()) / (s.std() or 1.0))
    return creatives, seconds


# ---------------------------------------------------------------- viewers --
def make_viewers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = np.clip(rng.normal(33, 12, n), 18, 72).round().astype(int)
    noisy = rng.choice([True, False], n, p=[0.28, 0.72])
    return pd.DataFrame({
        "viewer_id": [f"V{i:04d}" for i in range(1, n + 1)],
        "age": age,
        "age_band": pd.cut(age, [17, 24, 34, 44, 54, 100],
                           labels=["18-24", "25-34", "35-44", "45-54", "55+"]).astype(str),
        "gender": rng.choice(["female", "male", "non-binary"], n, p=[0.52, 0.45, 0.03]),
        "region": rng.choice(["UK", "US", "IT", "other"], n, p=[0.35, 0.35, 0.18, 0.12]),
        "income_band": rng.choice(["low", "medium", "high"], n, p=[0.3, 0.5, 0.2]),
        "device": rng.choice(["mobile", "desktop", "tablet"], n, p=[0.62, 0.30, 0.08]),
        "environment": np.where(noisy, "noisy", "quiet"),
        "recruitment_source": rng.choice(["panel", "social", "referral"], n, p=[0.6, 0.3, 0.1]),
        "consent_biometric": rng.random(n) > 0.04,
        # Latent traits: stable across this viewer's sessions.
        "trait_attention": rng.normal(0, 1, n).round(4),
        "trait_expressiveness": rng.normal(0, 1, n).round(4),
        "trait_acquiescence": rng.normal(0, 1, n).round(4),
        "latent_implicit_pref": rng.normal(config.IAT["pref_mean"],
                                           config.IAT["pref_sd"], n).round(4),
    })


# -------------------------------------------------------------------- IAT --
def make_iat(rng: np.random.Generator, viewers: pd.DataFrame
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reaction-time trials, then the D-score derived from them.

    The latent preference is the cause; the trials are a noisy measurement.
    Scoring recovers it imperfectly, exactly as a real IAT would.
    """
    p = config.IAT
    rows = []
    for v in viewers.itertuples():
        for block, congruent in (("congruent", True), ("incongruent", False)):
            n = p["trials_per_block"]
            # Incongruent pairings cost more time the stronger the preference.
            shift = 0.0 if congruent else p["incongruent_penalty"] * v.latent_implicit_pref
            log_rt = rng.normal(p["log_rt_mean"] + shift, p["log_rt_sd"], n)
            rt = np.exp(log_rt)
            rows.append(pd.DataFrame({
                "viewer_id": v.viewer_id,
                "block": block,
                "trial_index": np.arange(n),
                "is_congruent": congruent,
                "reaction_time_ms": rt.round(1),
                "is_error": rng.random(n) < p["error_rate"],
            }))
    trials = pd.concat(rows, ignore_index=True)

    # D-score, simplified Greenwald: error trials dropped, difference of means
    # over the pooled standard deviation.
    valid = trials[~trials.is_error]
    agg = valid.groupby(["viewer_id", "is_congruent"]).reaction_time_ms.mean().unstack()
    pooled = valid.groupby("viewer_id").reaction_time_ms.std()
    scores = pd.DataFrame({
        "viewer_id": agg.index,
        "mean_rt_congruent_ms": agg[True].round(1).values,
        "mean_rt_incongruent_ms": agg[False].round(1).values,
        "pooled_sd_ms": pooled.reindex(agg.index).round(1).values,
        "n_valid_trials": valid.groupby("viewer_id").size().reindex(agg.index).values,
    })
    scores["d_score"] = ((scores.mean_rt_incongruent_ms - scores.mean_rt_congruent_ms)
                         / scores.pooled_sd_ms).round(4)
    return trials, scores


# -------------------------------------------------- sessions + biometrics --
def make_sessions_and_biometrics(
    rng: np.random.Generator, viewers: pd.DataFrame,
    creatives: pd.DataFrame, cseconds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    A, V, D, S = config.ATTENTION, config.VALENCE, config.DROPOUT, config.SIGNAL
    by_creative = {cid: g.reset_index(drop=True) for cid, g in cseconds.groupby("creative_id")}
    ids = list(by_creative)
    lo, hi = config.CREATIVES_PER_VIEWER
    t0 = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)

    session_rows, bio_frames = [], []
    for v in viewers.itertuples():
        # Incomplete block design: each viewer sees a random subset.
        chosen = rng.choice(ids, size=rng.integers(lo, hi + 1), replace=False)
        for cid in chosen:
            f = by_creative[cid]
            n = len(f)
            session_id = f"S{len(session_rows) + 1:05d}"

            # --- attention: a logistic function of the creative, this second ---
            lin = (A["intercept"]
                   + A["position_in_ad"]  * f.position_in_ad.to_numpy()
                   + A["cut_density"]     * f.n_cuts.to_numpy()
                   + A["has_face"]        * f.has_face.to_numpy().astype(float)
                   + A["is_speech"]       * f.has_speech.to_numpy().astype(float)
                   + A["has_screen_text"] * f.has_screen_text.to_numpy().astype(float)
                   + A["motion_z"]        * f.motion_z.to_numpy()
                   + A["viewer_sd"]       * v.trait_attention
                   + _ar1(rng, n, A["ar1_rho"], A["noise_sd"]))
            attention = _sigmoid(lin)

            # --- abandonment: hazard rises as attention falls ---
            hazard = D["base_hazard"] + D["attention_weight"] * (1.0 - attention)
            quit_at = n
            draws = rng.random(n)
            left = np.nonzero(draws < hazard)[0]
            if left.size:
                quit_at = int(left[0]) + 1
            completed = quit_at >= n
            watched = f.iloc[:quit_at]
            attention = attention[:quit_at]

            session_rows.append({
                "session_id": session_id,
                "viewer_id": v.viewer_id,
                "creative_id": cid,
                "started_at": t0 + timedelta(minutes=int(rng.integers(0, 60 * 24 * 14))),
                "device": v.device,
                "environment": v.environment,
                "watch_time_s": int(quit_at),
                "creative_duration_s": int(n),
                "completed": bool(completed),
                "consent_biometric": bool(v.consent_biometric),
            })
            if not v.consent_biometric or quit_at == 0:
                continue   # no biometric rows without consent -- by design

            # --- valence: on-screen affect, music, and implicit preference ---
            val_lin = (V["intercept"]
                       + V["creative_happiness"] * watched.creative_happiness.to_numpy()
                       + V["music"]              * watched.is_music.to_numpy().astype(float)
                       + V["implicit_pref"]      * v.latent_implicit_pref
                       + V["viewer_sd"]          * v.trait_expressiveness
                       + _ar1(rng, quit_at, V["ar1_rho"], V["noise_sd"]))
            valence = np.clip(np.tanh(val_lin), -1, 1)

            # --- signal quality: the webcam does not always find the face ---
            p_found = np.clip(S["face_found_base"]
                              - S["face_found_attention"] * (1 - attention)
                              - (S["noisy_env_penalty"] if v.environment == "noisy" else 0),
                              0.5, 1.0)
            face_found = rng.random(quit_at) < p_found

            # --- expression: neutral-dominant, like the measurement model ---
            happy = _sigmoid(2.4 * valence - 0.9)
            sad = _sigmoid(-2.2 * valence - 1.7)
            surprise = 0.05 + 0.25 * watched.n_cuts.to_numpy().clip(0, 2) / 2
            small = rng.random((quit_at, 4)) * 0.05        # anger, disgust, fear, contempt
            raw = np.column_stack([np.full(quit_at, 1.0), happy, surprise, sad, small])
            probs = raw / raw.sum(axis=1, keepdims=True)
            probs[~face_found] = np.nan                    # NULL, not zero

            bio = pd.DataFrame({
                "session_id": session_id,
                "creative_id": cid,
                "viewer_id": v.viewer_id,
                "second": watched.second.to_numpy(),
                "attention_index": np.round(attention, 4),
                "gaze_on_screen": rng.random(quit_at) < attention,
                "face_detected": face_found,
                "valence": np.where(face_found, np.round(valence, 4), np.nan),
                "arousal": np.where(face_found,
                                    np.round(np.clip(0.3 + 0.4 * attention
                                                     + 0.2 * np.abs(valence), 0, 1), 4),
                                    np.nan),
            })
            for i, label in enumerate(config.EMOTIONS):
                bio[f"p_{label}"] = np.round(probs[:, i], 4)
            bio_frames.append(bio)

    return pd.DataFrame(session_rows), pd.concat(bio_frames, ignore_index=True)


# ----------------------------------------------------------------- survey --
QUESTIONS = [
    ("q_ad_recall",       "Do you remember seeing this ad?",             "binary"),
    ("q_brand_recall",    "Which brand was being advertised?",           "binary"),
    ("q_likeability",     "How much did you like the ad?",               "likert_5"),
    ("q_message_clarity", "How clear was the main message?",             "likert_5"),
    ("q_brand_fit",       "How well did the ad fit the brand?",          "likert_5"),
    ("q_purchase_intent", "How likely are you to consider this brand?",  "likert_5"),
    ("q_would_share",     "How likely are you to share this ad?",        "likert_5"),
    ("q_emotional",       "How emotionally affecting was the ad?",       "likert_5"),
]


def make_survey(rng: np.random.Generator, sessions: pd.DataFrame,
                bio: pd.DataFrame, viewers: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Answers follow from what happened during the exposure."""
    C = config.SURVEY
    per_session = (bio.groupby("session_id")
                      .agg(mean_attention=("attention_index", "mean"),
                           mean_valence=("valence", "mean"))
                      .reset_index())
    s = (sessions.merge(per_session, on="session_id", how="left")
                 .merge(viewers[["viewer_id", "trait_acquiescence",
                                 "latent_implicit_pref"]], on="viewer_id"))
    s[["mean_attention", "mean_valence"]] = s[["mean_attention", "mean_valence"]].fillna(0.0)

    latent = (C["intercept"]
              + C["mean_attention"] * s.mean_attention
              + C["mean_valence"]   * s.mean_valence
              + C["completed"]      * s.completed.astype(float)
              + C["implicit_pref"]  * s.latent_implicit_pref
              + C["acquiescence_sd"] * s.trait_acquiescence).to_numpy()

    rows = []
    for qid, _, rtype in QUESTIONS:
        draw = latent + rng.normal(0, C["noise_sd"], len(s))
        if rtype == "likert_5":
            value = np.clip(np.round(draw), 1, 5).astype(int)
        else:
            value = (_sigmoid(draw - 2.5) > rng.random(len(s))).astype(int)
        rows.append(pd.DataFrame({
            "session_id": s.session_id, "viewer_id": s.viewer_id,
            "creative_id": s.creative_id, "question_id": qid,
            "response_type": rtype, "value_numeric": value,
        }))
    questions = pd.DataFrame(QUESTIONS, columns=["question_id", "question_text", "response_type"])
    return questions, pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------- the params --
def make_params_table() -> pd.DataFrame:
    rows = []
    for component, block in (("attention", config.ATTENTION), ("valence", config.VALENCE),
                             ("survey", config.SURVEY), ("iat", config.IAT),
                             ("dropout", config.DROPOUT), ("signal", config.SIGNAL)):
        for parameter, value in block.items():
            rows.append({"component": component, "parameter": parameter,
                         "value": float(value)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ build --
def generate(seed: int | None = None, n_viewers: int | None = None) -> dict[str, pd.DataFrame]:
    seed = config.SEED if seed is None else seed
    n_viewers = config.N_VIEWERS if n_viewers is None else n_viewers

    # One seed, independent streams per section: adding a section cannot change
    # the numbers drawn by an earlier one.
    streams = dict(zip(RNG_SECTIONS, np.random.default_rng(seed).spawn(len(RNG_SECTIONS))))

    creatives, cseconds = load_creative_features()
    viewers = make_viewers(streams["viewers"], n_viewers)
    trials, scores = make_iat(streams["iat"], viewers)
    sessions, bio = make_sessions_and_biometrics(streams["sessions"], viewers,
                                                 creatives, cseconds)
    questions, responses = make_survey(streams["survey"], sessions, bio, viewers)

    tables = {
        "viewers": viewers, "sessions": sessions, "biometric_seconds": bio,
        "survey_questions": questions, "survey_responses": responses,
        "iat_trials": trials, "iat_scores": scores,
        "generator_params": make_params_table(),
    }
    tables["generator_manifest"] = pd.DataFrame([{
        "run_id": uuid.uuid4().hex[:12],
        "generator_version": config.GENERATOR_VERSION,
        "seed": seed,
        "n_viewers": n_viewers,
        "n_sessions": len(sessions),
        "n_biometric_rows": len(bio),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }])
    # Every audience row is stamped, so a stray join can never hide its origin.
    for name, df in tables.items():
        df.insert(len(df.columns), "data_source", config.DATA_SOURCE)
    return tables
