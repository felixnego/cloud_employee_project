"""Panel size, seed, and the ground-truth coefficients of the generator.

Every number in this file is published to `audience.generator_params` so that
analysis can be checked against the truth that produced the data: if a
regression on the synthetic panel does not recover `ATTENTION["cut_density"]`,
the analysis is wrong, not the data.
"""
from __future__ import annotations

import os
from pathlib import Path

from pipeline import config as creative

# Audience depends on the creative layer; never the reverse.
WAREHOUSE_DIR = creative.WAREHOUSE_DIR
AUDIENCE_DIR = WAREHOUSE_DIR / "audience"
DUCKDB_PATH = creative.DUCKDB_PATH
TRANSFORM_DIR = Path(os.environ.get("AUDIENCE_TRANSFORM_DIR", "transforms_audience")).resolve()

EMOTIONS = creative.EMOTION_LABELS          # same 8 classes as the creative side

GENERATOR_VERSION = "1.0"
SEED = int(os.environ.get("AUDIENCE_SEED", "42"))
N_VIEWERS = int(os.environ.get("AUDIENCE_N_VIEWERS", "300"))
CREATIVES_PER_VIEWER = (2, 3)               # incomplete block design
DATA_SOURCE = "synthetic_v1"                # stamped on every audience row

# ----------------------------------------------------------- ground truth --
# Attention is a logistic function of what the creative is doing at that second.
# These are the effects an analyst should be able to recover.
ATTENTION = {
    "intercept":        1.10,
    "position_in_ad":  -1.30,   # attention decays across the runtime
    "cut_density":      0.45,   # a cut re-captures attention
    "has_face":         0.35,   # faces hold the eye
    "is_speech":        0.25,
    "has_screen_text": -0.15,   # reading competes with watching
    "motion_z":         0.20,
    "viewer_sd":        0.60,   # between-viewer variation
    "ar1_rho":          0.70,   # responses are smooth, not independent
    "noise_sd":         0.35,
}

# Valence: how positive the viewer's expression is.
VALENCE = {
    "intercept":           0.05,
    "creative_happiness":  0.55,   # smiling actors -> smiling viewers
    "music":               0.15,
    "implicit_pref":       0.30,   # IAT latent trait shows up on the face
    "viewer_sd":           0.25,
    "ar1_rho":             0.65,
    "noise_sd":            0.25,
}

# Likert answers (1-5) as a function of the session that produced them.
SURVEY = {
    "intercept":        1.15,   # keeps Likert answers off the ceiling
    "mean_attention":   1.90,
    "mean_valence":     0.90,
    "completed":        0.40,
    "implicit_pref":    0.50,
    "acquiescence_sd":  0.45,   # some people just answer high
    "noise_sd":         0.55,
}

# Implicit Association Test. Congruent trials are faster than incongruent ones
# by an amount proportional to the viewer's latent implicit preference.
IAT = {
    "trials_per_block":     20,
    "log_rt_mean":          6.55,   # exp(6.55) ~ 700 ms
    "log_rt_sd":            0.32,
    "incongruent_penalty":  0.18,   # in log-ms, scaled by implicit preference
    "error_rate":           0.07,
    "pref_mean":            0.35,   # literature-typical D-score centre
    "pref_sd":              0.40,
}

# Abandonment. Hazard rises as attention falls.
DROPOUT = {
    "base_hazard":      0.0030,
    "attention_weight": 0.0320,
}

# Webcam reliability: the face is not always found.
SIGNAL = {
    "face_found_base":      0.97,
    "face_found_attention": 0.05,   # looking away -> more missing frames
    "noisy_env_penalty":    0.04,
}
