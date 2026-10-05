"""The audience data contract.

These seven tables are what the rest of the project agrees to read. The
synthetic generator fills them today; a real ingest from a biometric vendor, a
survey platform and an IAT tool would fill the same tables tomorrow, and
nothing downstream would change.

Column *types* are introspected from the warehouse at documentation time (see
`pipeline/docgen.py`), so only the things a tool cannot infer live here: the
grain, the key, what the table is for, and where it will bite.
"""
from __future__ import annotations

# Order matters only for presentation.
TABLES: dict[str, dict] = {
    "viewers": dict(
        grain="one row per panelist",
        key="viewer_id",
        purpose="The panel. Demographics, device, consent, and two latent traits "
                "(attentiveness, expressiveness) that persist across sessions.",
        notes=["`latent_implicit_pref` is the viewer's true implicit preference. "
               "It is deliberately exposed: the IAT D-score is a *noisy "
               "measurement* of it, so analysis can be checked against truth. "
               "No real dataset has this column.",
               "`consent_biometric` is false for a few panelists, whose sessions "
               "carry no biometric rows. Consent is modelled even in synthetic "
               "form because the real pipeline must honour it."],
    ),
    "sessions": dict(
        grain="one row per (viewer, creative) exposure",
        key="session_id",
        purpose="The exposure fact: who watched what, on what device, in what "
                "environment, and whether they finished.",
        notes=["Not every viewer sees every creative -- an incomplete block "
               "design, as a real panel would be. Check cell sizes before "
               "comparing creatives on a subgroup.",
               "`watch_time_s` is shorter than the creative's duration when the "
               "viewer abandoned; `completed` flags it."],
    ),
    "biometric_seconds": dict(
        grain="one row per (session, second)",
        key="session_id, second",
        purpose="The webcam-derived response trace. Joins directly onto "
                "`main.creative_seconds` on (creative_id, second).",
        notes=["The eight `p_*` emotion columns use the same taxonomy as the "
               "creative side's `face_detections`, so viewer expression and "
               "on-screen expression are directly comparable.",
               "When `face_detected` is false the emotion and valence columns "
               "are NULL, not zero -- the webcam lost the face. Filter, do not "
               "impute, unless the analysis says otherwise.",
               "Rows stop at abandonment. Absence of a row is data: it means the "
               "viewer had already left."],
    ),
    "survey_questions": dict(
        grain="one row per question",
        key="question_id",
        purpose="The post-exposure battery: recall, likeability, comprehension, "
                "purchase intent.",
        notes=["Long format on purpose -- adding a question is a row, not a "
               "schema migration."],
    ),
    "survey_responses": dict(
        grain="one row per (session, question)",
        key="session_id, question_id",
        purpose="Answers to the battery, driven by what actually happened during "
                "the exposure.",
        notes=["Response styles vary by panelist: some answer consistently high "
               "(acquiescence). `value_numeric` is comparable across questions "
               "of the same `response_type` only."],
    ),
    "iat_trials": dict(
        grain="one row per (viewer, block, trial)",
        key="viewer_id, block, trial_index",
        purpose="Raw Implicit Association Test reaction times -- the measurement, "
                "before scoring.",
        notes=["Congruent trials are faster than incongruent ones in proportion "
               "to the viewer's implicit preference. Error trials are flagged "
               "and excluded by the scoring step."],
    ),
    "iat_scores": dict(
        grain="one row per viewer",
        key="viewer_id",
        purpose="The derived D-score: (mean incongruent RT - mean congruent RT) "
                "divided by the pooled standard deviation.",
        notes=["Positive D means faster on congruent pairings, i.e. an implicit "
               "preference toward the advertised brand.",
               "`d_score` should correlate strongly but imperfectly with "
               "`viewers.latent_implicit_pref` -- that gap is measurement error, "
               "and it is intentional."],
    ),
    "generator_params": dict(
        grain="one row per parameter",
        key="component, parameter",
        purpose="The ground-truth coefficients used to generate this panel.",
        notes=["This is the point of a synthetic dataset: analysis can be "
               "validated. A model fitted to `analysis.viewer_seconds` should "
               "recover these values. If it does not, suspect the analysis.",
               "A real panel has no such table."],
    ),
    "generator_manifest": dict(
        grain="one row per generation run",
        key="run_id",
        purpose="Provenance: seed, generator version, panel size, row counts.",
        notes=["The seed makes a run byte-reproducible: same seed, same panel."],
    ),
}

ANALYSIS_TABLES: dict[str, dict] = {
    "viewer_seconds": dict(
        grain="one row per (session, second)",
        key="session_id, second",
        purpose="The joined table the whole project exists to produce: each "
                "second of each viewer's response, alongside what the creative "
                "was doing at that second and who the viewer is.",
        notes=["MIXED PROVENANCE. Creative columns are measured from real video "
               "files; viewer columns are synthetic. `data_source` marks it.",
               "This is the table to model on. One row per viewer-second, every "
               "predictor already attached."],
    ),
    "creative_performance": dict(
        grain="one row per creative",
        key="creative_id",
        purpose="Per-creative rollup of the synthetic panel: completion, "
                "attention, valence and survey scores next to the creative's own "
                "measured attributes.",
        notes=["Differences between creatives here are generator artefacts, not "
               "findings about these ads. The creative-side columns are real; "
               "the response columns are not."],
    ),
}
