-- ---------------------------------------------------------------------------
-- The analysis schema: where measured creative features meet synthetic
-- response. MIXED PROVENANCE -- creative columns are real, viewer columns are
-- generated. Kept in its own schema so that is never in doubt.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS analysis;

-- The table the whole project exists to produce: one row per viewer-second,
-- with every predictor already attached. This is the shape a real biometric
-- export would land in too.
CREATE OR REPLACE TABLE analysis.viewer_seconds AS
SELECT
    b.session_id,
    b.viewer_id,
    b.creative_id,
    b.second,

    -- response (synthetic)
    b.attention_index,
    b.gaze_on_screen,
    b.face_detected,
    b.valence,
    b.arousal,
    b.p_happiness       AS viewer_p_happiness,
    b.p_neutral         AS viewer_p_neutral,
    b.p_surprise        AS viewer_p_surprise,
    b.p_sadness         AS viewer_p_sadness,

    -- what the creative was doing at that second (measured)
    cs.position_in_ad,
    cs.n_cuts,
    cs.motion,
    cs.brightness,
    cs.audio_class,
    cs.has_speech,
    cs.has_face         AS creative_has_face,
    cs.has_screen_text,
    cs.mean_p_happiness AS creative_p_happiness,
    cs.modal_shot_index,

    -- who is watching (synthetic)
    v.age_band,
    v.gender,
    v.region,
    v.device,
    v.environment,
    v.latent_implicit_pref,
    i.d_score           AS iat_d_score,

    -- exposure context (synthetic)
    s.completed,
    s.watch_time_s,

    'creative=measured; viewer=synthetic_v1' AS data_source
FROM audience.biometric_seconds b
JOIN audience.sessions  s USING (session_id)
JOIN audience.viewers   v ON v.viewer_id = b.viewer_id
LEFT JOIN audience.iat_scores i ON i.viewer_id = b.viewer_id
JOIN main.creative_seconds cs
  ON cs.creative_id = b.creative_id
 AND cs.second      = b.second
ORDER BY b.session_id, b.second;

-- Per-creative rollup. The creative columns are real; the response columns are
-- generator output and say nothing about these ads in the world.
CREATE OR REPLACE TABLE analysis.creative_performance AS
WITH response AS (
    SELECT
        creative_id,
        count(DISTINCT session_id)                       AS n_sessions_with_biometrics,
        round(avg(attention_index), 4)                   AS mean_attention,
        round(avg(valence), 4)                           AS mean_valence,
        round(avg(CASE WHEN gaze_on_screen THEN 1.0 ELSE 0 END), 4) AS gaze_share,
        round(avg(CASE WHEN face_detected  THEN 1.0 ELSE 0 END), 4) AS face_found_share
    FROM analysis.viewer_seconds
    GROUP BY creative_id
),
completion AS (
    SELECT creative_id,
           count(*)                                               AS n_sessions,
           round(avg(CASE WHEN completed THEN 1.0 ELSE 0 END), 4) AS completion_rate,
           round(avg(watch_time_s::DOUBLE / nullif(creative_duration_s, 0)), 4) AS mean_watched_share
    FROM audience.sessions
    GROUP BY creative_id
),
survey AS (
    SELECT creative_id,
           round(avg(value_numeric) FILTER (WHERE question_id = 'q_likeability'), 3)     AS likeability,
           round(avg(value_numeric) FILTER (WHERE question_id = 'q_purchase_intent'), 3) AS purchase_intent,
           round(avg(value_numeric) FILTER (WHERE question_id = 'q_message_clarity'), 3) AS message_clarity,
           round(avg(value_numeric) FILTER (WHERE question_id = 'q_brand_recall'), 3)    AS brand_recall_rate
    FROM audience.survey_responses
    GROUP BY creative_id
)
SELECT
    c.creative_id,
    c.display_name,
    -- measured
    c.duration_s, c.n_shots, c.cuts_per_minute, c.words_per_minute,
    c.modal_setting, c.modal_style,
    -- synthetic
    cp.n_sessions, r.n_sessions_with_biometrics,
    cp.completion_rate, cp.mean_watched_share,
    r.mean_attention, r.mean_valence, r.gaze_share, r.face_found_share,
    sv.likeability, sv.purchase_intent, sv.message_clarity, sv.brand_recall_rate,
    'creative=measured; response=synthetic_v1' AS data_source
FROM main.creative_summary c
LEFT JOIN response   r  USING (creative_id)
LEFT JOIN completion cp USING (creative_id)
LEFT JOIN survey     sv USING (creative_id)
ORDER BY c.creative_id;

COPY audience.viewers            TO '${MARTS}/audience_viewers.parquet'            (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.sessions           TO '${MARTS}/audience_sessions.parquet'           (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.biometric_seconds  TO '${MARTS}/audience_biometric_seconds.parquet'  (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.survey_questions   TO '${MARTS}/audience_survey_questions.parquet'   (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.survey_responses   TO '${MARTS}/audience_survey_responses.parquet'   (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.iat_trials         TO '${MARTS}/audience_iat_trials.parquet'         (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.iat_scores         TO '${MARTS}/audience_iat_scores.parquet'         (FORMAT PARQUET, COMPRESSION ZSTD);
COPY audience.generator_params   TO '${MARTS}/audience_generator_params.parquet'   (FORMAT PARQUET, COMPRESSION ZSTD);

COPY analysis.viewer_seconds       TO '${MARTS}/viewer_seconds.parquet'       (FORMAT PARQUET, COMPRESSION ZSTD);
COPY analysis.creative_performance TO '${MARTS}/creative_performance.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY analysis.creative_performance TO '${MARTS}/creative_performance.csv'     (FORMAT CSV, HEADER);
