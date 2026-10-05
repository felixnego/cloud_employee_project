-- ---------------------------------------------------------------------------
-- Layer 3: THE SPINE.
--
-- One row per (creative, sampled frame) at 4 Hz, carrying every signal that
-- varies over time. This is the table audience data is designed to join onto:
-- biometric traces arrive as (viewer_id, creative_id, t) and land here with a
-- single equi-join once t is snapped to the frame grid -- no interpolation,
-- no window functions, no per-analyst resampling logic.
--
-- NULL vs zero matters here. `n_objects` is NULL on frames YOLOX never looked
-- at (it samples at 2 Hz) and 0 on frames it looked at and found nothing. The
-- `*_sampled` flags make that distinction explicit instead of leaving it to be
-- rediscovered by whoever writes the first AVG().
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE creative_frames AS
WITH cfg AS (
    SELECT
        creative_id,
        max(step_frames) FILTER (WHERE extractor = 'objects') AS objects_step,
        max(step_frames) FILTER (WHERE extractor = 'ocr')     AS ocr_step,
        max(step_frames) FILTER (WHERE extractor = 'faces')   AS faces_step
    FROM raw_sampling_config
    GROUP BY creative_id
),
frame_shot AS (
    -- ASOF join: for each frame, the most recent shot that had started.
    -- Cheaper and more robust than a BETWEEN range join, and it cannot leave
    -- a frame without a shot the way a half-open interval can at the tail.
    SELECT
        v.creative_id,
        v.frame_index,
        sh.shot_index,
        sh.start_s    AS shot_start_s,
        sh.duration_s AS shot_duration_s
    FROM raw_visual_frames v
    ASOF LEFT JOIN raw_shot_boundaries sh
      ON sh.creative_id = v.creative_id
     AND v.t_start_s >= sh.start_s
),
faces_agg AS (
    SELECT
        creative_id,
        frame_index,
        count(*)                                                AS n_faces,
        round(max(area_frac), 5)                                AS largest_face_area_frac,
        any_value(emotion_top)      FILTER (WHERE is_largest)   AS largest_face_emotion,
        any_value(emotion_top_prob) FILTER (WHERE is_largest)   AS largest_face_emotion_prob,
        round(avg(p_neutral), 4)                                AS mean_p_neutral,
        round(avg(p_happiness), 4)                              AS mean_p_happiness,
        round(avg(p_surprise), 4)                               AS mean_p_surprise,
        round(avg(p_sadness), 4)                                AS mean_p_sadness
    FROM raw_face_detections
    GROUP BY creative_id, frame_index
),
objects_agg AS (
    SELECT
        creative_id,
        frame_index,
        count(*)                                     AS n_objects,
        count(*) FILTER (WHERE class_name = 'person') AS n_persons,
        string_agg(DISTINCT class_name, ', ')        AS object_classes,
        round(max(area_frac), 5)                     AS largest_object_area_frac
    FROM raw_object_detections
    GROUP BY creative_id, frame_index
),
text_agg AS (
    SELECT
        creative_id,
        frame_index,
        count(*)                      AS n_text_detections,
        round(max(height_frac), 4)    AS max_text_height_frac,
        string_agg(text, ' | ')       AS screen_text
    FROM raw_screen_text_detections
    GROUP BY creative_id, frame_index
),
speech_frames AS (
    -- A frame counts as speech when its window overlaps a Whisper segment.
    -- Whisper runs with Silero VAD, so this is model-grounded, not a heuristic.
    SELECT DISTINCT v.creative_id, v.frame_index
    FROM raw_visual_frames v
    JOIN raw_transcript_segments s
      ON s.creative_id = v.creative_id
     AND v.t_start_s < s.end_s
     AND v.t_end_s   > s.start_s
)
SELECT
    v.creative_id,
    v.frame_index,
    v.t_start_s,
    v.t_end_s,
    round(v.t_start_s / nullif(c.duration_s, 0), 5)        AS position_in_ad,

    -- shot context
    fs.shot_index,
    round(v.t_start_s - fs.shot_start_s, 3)                AS t_in_shot_s,
    row_number() OVER (PARTITION BY v.creative_id, fs.shot_index
                       ORDER BY v.frame_index) = 1         AS is_shot_start,
    (row_number() OVER (PARTITION BY v.creative_id, fs.shot_index
                        ORDER BY v.frame_index) = 1
       AND fs.shot_index > 0)                              AS is_cut,

    -- feature 4: visual
    v.brightness, v.brightness_std, v.saturation, v.contrast,
    v.colourfulness, v.sharpness, v.edge_density, v.dominant_hue_deg,
    v.motion_abs_diff, v.hist_distance, v.is_near_black,

    -- feature 5: audio
    a.dbfs, a.rms, a.spectral_centroid_hz, a.spectral_flatness,
    a.spectral_rolloff_hz, a.zero_crossing_rate, a.onset_strength,
    a.harmonic_ratio, a.is_silent,

    -- feature 3: speech presence
    (sp.frame_index IS NOT NULL)                           AS is_speech,

    -- feature 7: faces (sampled every frame)
    coalesce(fa.n_faces, 0)                                AS n_faces,
    fa.largest_face_area_frac,
    fa.largest_face_emotion,
    fa.largest_face_emotion_prob,
    fa.mean_p_neutral, fa.mean_p_happiness,
    fa.mean_p_surprise, fa.mean_p_sadness,

    -- feature 8: objects (subsampled -- see objects_sampled)
    ((v.frame_index - 1) % cfg.objects_step = 0)           AS objects_sampled,
    CASE WHEN (v.frame_index - 1) % cfg.objects_step = 0
         THEN coalesce(ob.n_objects, 0) END                AS n_objects,
    CASE WHEN (v.frame_index - 1) % cfg.objects_step = 0
         THEN coalesce(ob.n_persons, 0) END                AS n_persons,
    ob.object_classes,
    ob.largest_object_area_frac,

    -- feature 6: on-screen text (subsampled -- see ocr_sampled)
    ((v.frame_index - 1) % cfg.ocr_step = 0)               AS ocr_sampled,
    CASE WHEN (v.frame_index - 1) % cfg.ocr_step = 0
         THEN coalesce(tx.n_text_detections, 0) END        AS n_text_detections,
    tx.max_text_height_frac,
    tx.screen_text
FROM raw_visual_frames v
JOIN creatives     c  USING (creative_id)
JOIN cfg           cfg USING (creative_id)
LEFT JOIN frame_shot   fs USING (creative_id, frame_index)
LEFT JOIN raw_audio_frames a USING (creative_id, frame_index)
LEFT JOIN faces_agg    fa USING (creative_id, frame_index)
LEFT JOIN objects_agg  ob USING (creative_id, frame_index)
LEFT JOIN text_agg     tx USING (creative_id, frame_index)
LEFT JOIN speech_frames sp USING (creative_id, frame_index)
ORDER BY v.creative_id, v.frame_index;
