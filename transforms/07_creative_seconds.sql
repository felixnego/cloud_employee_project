-- ---------------------------------------------------------------------------
-- Layer 3: the analyst-facing spine, one row per (creative, second).
--
-- Same information as `creative_frames`, aggregated to whole seconds. This is
-- the table to start from: survey timestamps, dial-test traces and most
-- biometric exports arrive at 1 Hz, and "what was on screen at 0:14" is a
-- question about a second, not about a 250 ms frame. Drop to
-- `creative_frames` when a sub-second reveal matters.
--
-- `audio_class` is the one derived field with judgement in it. Speech comes
-- from Whisper's VAD-backed segments (model-grounded); music vs ambient is an
-- explicit threshold on harmonic ratio and spectral flatness, stated in the
-- `params` CTE so it can be argued with and changed in one place.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE creative_seconds AS
WITH params AS (
    SELECT
        -50.0 AS silence_dbfs,       -- below this, treat the second as silent
        0.55  AS music_harmonic_min, -- tonal content dominates -> music-like
        0.02  AS music_flatness_max  -- low flatness -> pitched, not noise
),
agg AS (
    SELECT
        f.creative_id,
        floor(f.t_start_s)::INTEGER           AS second,
        count(*)                              AS n_frames,

        -- shot / pacing
        min(f.shot_index)                     AS first_shot_index,
        mode(f.shot_index)                    AS modal_shot_index,
        count(*) FILTER (WHERE f.is_cut)      AS n_cuts,

        -- visual
        round(avg(f.brightness), 4)           AS brightness,
        round(avg(f.saturation), 4)           AS saturation,
        round(avg(f.contrast), 4)             AS contrast,
        round(avg(f.colourfulness), 3)        AS colourfulness,
        round(avg(f.edge_density), 4)         AS edge_density,
        round(avg(f.motion_abs_diff), 5)      AS motion,
        round(max(f.motion_abs_diff), 5)      AS peak_motion,
        round(max(f.hist_distance), 5)        AS peak_hist_distance,

        -- audio
        round(avg(f.dbfs), 2)                 AS dbfs,
        round(max(f.dbfs), 2)                 AS peak_dbfs,
        round(avg(f.spectral_centroid_hz), 1) AS spectral_centroid_hz,
        round(avg(f.spectral_flatness), 5)    AS spectral_flatness,
        round(avg(f.onset_strength), 3)       AS onset_strength,
        round(avg(f.harmonic_ratio), 4)       AS harmonic_ratio,

        -- speech
        bool_or(f.is_speech)                  AS has_speech,
        round(avg(CASE WHEN f.is_speech THEN 1.0 ELSE 0.0 END), 3) AS speech_share,

        -- faces
        max(f.n_faces)                        AS max_faces,
        round(avg(f.n_faces), 3)              AS mean_faces,
        round(max(f.largest_face_area_frac), 5) AS largest_face_area_frac,
        mode(f.largest_face_emotion)          AS modal_face_emotion,
        round(avg(f.mean_p_happiness), 4)     AS mean_p_happiness,
        round(avg(f.mean_p_surprise), 4)      AS mean_p_surprise,
        round(avg(f.mean_p_sadness), 4)       AS mean_p_sadness,

        -- objects (only frames YOLOX actually saw)
        max(f.n_objects)                      AS max_objects,
        max(f.n_persons)                      AS max_persons,
        string_agg(DISTINCT f.object_classes, ', ') AS object_classes,

        -- on-screen text (only frames OCR actually saw)
        max(f.n_text_detections)              AS max_text_detections,
        round(max(f.max_text_height_frac), 4) AS max_text_height_frac,
        any_value(f.screen_text)              AS screen_text
    FROM creative_frames f
    GROUP BY f.creative_id, floor(f.t_start_s)::INTEGER
)
SELECT
    a.creative_id,
    a.second,
    a.second::DOUBLE                                  AS t_start_s,
    (a.second + 1)::DOUBLE                            AS t_end_s,
    round(a.second / nullif(c.duration_s, 0), 4)      AS position_in_ad,
    a.n_frames,

    a.first_shot_index,
    a.modal_shot_index,
    a.n_cuts,

    a.brightness, a.saturation, a.contrast, a.colourfulness, a.edge_density,
    a.motion, a.peak_motion, a.peak_hist_distance,

    a.dbfs, a.peak_dbfs, a.spectral_centroid_hz, a.spectral_flatness,
    a.onset_strength, a.harmonic_ratio,

    a.has_speech, a.speech_share,

    -- The only judgement call in this table; thresholds live in `params`.
    CASE
        WHEN a.dbfs IS NULL                     THEN 'no_audio'
        WHEN a.dbfs < p.silence_dbfs            THEN 'silence'
        WHEN a.has_speech
             AND a.harmonic_ratio >= p.music_harmonic_min THEN 'speech_over_music'
        WHEN a.has_speech                       THEN 'speech'
        WHEN a.harmonic_ratio >= p.music_harmonic_min
             AND a.spectral_flatness <= p.music_flatness_max THEN 'music'
        ELSE 'ambient'
    END                                               AS audio_class,

    a.max_faces, a.mean_faces, a.largest_face_area_frac, a.modal_face_emotion,
    a.mean_p_happiness, a.mean_p_surprise, a.mean_p_sadness,
    (a.max_faces > 0)                                 AS has_face,

    a.max_objects, a.max_persons, a.object_classes,

    a.max_text_detections, a.max_text_height_frac, a.screen_text,
    (coalesce(a.max_text_detections, 0) > 0)          AS has_screen_text
FROM agg a
CROSS JOIN params p
JOIN creatives c USING (creative_id)
ORDER BY a.creative_id, a.second;
