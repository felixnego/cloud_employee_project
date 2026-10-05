-- ---------------------------------------------------------------------------
-- Layer 2: the shot dimension -- the unit a creative director actually thinks
-- in. Each shot carries its CLIP labels (pivoted from long to wide, because
-- "show me every product close-up" should not require an analyst to filter on
-- an axis column) plus what was visible and audible inside it.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE shots AS
WITH top_labels AS (
    SELECT
        creative_id,
        shot_index,
        max(label) FILTER (WHERE axis = 'shot_scale') AS label_shot_scale,
        max(score) FILTER (WHERE axis = 'shot_scale') AS conf_shot_scale,
        max(label) FILTER (WHERE axis = 'setting')    AS label_setting,
        max(score) FILTER (WHERE axis = 'setting')    AS conf_setting,
        max(label) FILTER (WHERE axis = 'subject')    AS label_subject,
        max(score) FILTER (WHERE axis = 'subject')    AS conf_subject,
        max(label) FILTER (WHERE axis = 'style')      AS label_style,
        max(score) FILTER (WHERE axis = 'style')      AS conf_style
    FROM raw_shot_label_scores
    WHERE is_top
    GROUP BY creative_id, shot_index
),
shot_visual AS (
    SELECT
        b.creative_id,
        b.shot_index,
        count(*)                                  AS n_sampled_frames,
        round(avg(v.brightness), 4)               AS mean_brightness,
        round(avg(v.saturation), 4)               AS mean_saturation,
        round(avg(v.contrast), 4)                 AS mean_contrast,
        round(avg(v.colourfulness), 3)            AS mean_colourfulness,
        round(avg(v.edge_density), 4)             AS mean_edge_density,
        round(avg(v.motion_abs_diff), 5)          AS mean_motion,
        round(max(v.motion_abs_diff), 5)          AS peak_motion,
        round(avg(v.dominant_hue_deg), 1)         AS mean_dominant_hue_deg
    FROM raw_shot_boundaries b
    JOIN raw_visual_frames v
      ON v.creative_id = b.creative_id
     AND v.t_start_s >= b.start_s
     AND v.t_start_s <  b.end_s
    GROUP BY b.creative_id, b.shot_index
),
shot_audio AS (
    SELECT
        b.creative_id,
        b.shot_index,
        round(avg(a.dbfs), 2)              AS mean_dbfs,
        round(avg(a.harmonic_ratio), 4)    AS mean_harmonic_ratio,
        round(avg(a.onset_strength), 3)    AS mean_onset_strength
    FROM raw_shot_boundaries b
    JOIN raw_audio_frames a
      ON a.creative_id = b.creative_id
     AND a.t_start_s >= b.start_s
     AND a.t_start_s <  b.end_s
    GROUP BY b.creative_id, b.shot_index
),
shot_faces AS (
    SELECT
        b.creative_id,
        b.shot_index,
        count(DISTINCT f.frame_index)                       AS n_frames_with_face,
        round(avg(f.area_frac), 5)                          AS mean_face_area_frac,
        round(max(f.area_frac), 5)                          AS max_face_area_frac,
        mode(f.emotion_top) FILTER (WHERE f.is_largest)     AS modal_face_emotion
    FROM raw_shot_boundaries b
    JOIN raw_face_detections f
      ON f.creative_id = b.creative_id
     AND f.t_start_s >= b.start_s
     AND f.t_start_s <  b.end_s
    GROUP BY b.creative_id, b.shot_index
),
shot_objects AS (
    SELECT
        b.creative_id,
        b.shot_index,
        string_agg(DISTINCT o.class_name, ', ')  AS object_classes,
        count(DISTINCT o.class_name)             AS n_object_classes
    FROM raw_shot_boundaries b
    JOIN raw_object_detections o
      ON o.creative_id = b.creative_id
     AND o.t_start_s >= b.start_s
     AND o.t_start_s <  b.end_s
    GROUP BY b.creative_id, b.shot_index
),
shot_text AS (
    SELECT
        b.creative_id,
        b.shot_index,
        count(*)                               AS n_text_detections,
        round(max(t.height_frac), 4)           AS max_text_height_frac
    FROM raw_shot_boundaries b
    JOIN raw_screen_text_detections t
      ON t.creative_id = b.creative_id
     AND t.t_start_s >= b.start_s
     AND t.t_start_s <  b.end_s
    GROUP BY b.creative_id, b.shot_index
)
SELECT
    b.creative_id,
    b.shot_index,
    b.start_s,
    b.end_s,
    b.duration_s,
    -- Where in the ad this shot sits: lets you compare openings to end-frames
    -- across creatives of different lengths.
    round(b.start_s / nullif(c.duration_s, 0), 4) AS position_in_ad,
    (b.shot_index = 0)                            AS is_first_shot,
    (b.shot_index = c.n_shots - 1)                AS is_last_shot,

    l.label_shot_scale, l.conf_shot_scale,
    l.label_setting,    l.conf_setting,
    l.label_subject,    l.conf_subject,
    l.label_style,      l.conf_style,

    v.n_sampled_frames,
    v.mean_brightness, v.mean_saturation, v.mean_contrast,
    v.mean_colourfulness, v.mean_edge_density,
    v.mean_motion, v.peak_motion, v.mean_dominant_hue_deg,

    a.mean_dbfs, a.mean_harmonic_ratio, a.mean_onset_strength,

    coalesce(f.n_frames_with_face, 0)            AS n_frames_with_face,
    f.mean_face_area_frac,
    f.max_face_area_frac,
    f.modal_face_emotion,

    coalesce(o.n_object_classes, 0)              AS n_object_classes,
    o.object_classes,

    coalesce(t.n_text_detections, 0)             AS n_text_detections,
    t.max_text_height_frac
FROM raw_shot_boundaries b
JOIN creatives     c USING (creative_id)
LEFT JOIN top_labels   l USING (creative_id, shot_index)
LEFT JOIN shot_visual  v USING (creative_id, shot_index)
LEFT JOIN shot_audio   a USING (creative_id, shot_index)
LEFT JOIN shot_faces   f USING (creative_id, shot_index)
LEFT JOIN shot_objects o USING (creative_id, shot_index)
LEFT JOIN shot_text    t USING (creative_id, shot_index)
ORDER BY b.creative_id, b.shot_index;
