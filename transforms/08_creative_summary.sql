-- ---------------------------------------------------------------------------
-- Layer 3: one wide row per creative -- the table to open first.
--
-- Everything in here is derivable from the other marts; it exists so the
-- obvious comparison ("which of these five ads is fastest cut, most spoken,
-- most face-forward?") is a SELECT * rather than five CTEs.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE creative_summary AS
WITH second_rollup AS (
    SELECT
        creative_id,
        count(*)                                                   AS n_seconds,
        round(avg(brightness), 4)                                  AS mean_brightness,
        round(avg(saturation), 4)                                  AS mean_saturation,
        round(avg(motion), 5)                                      AS mean_motion,
        round(avg(colourfulness), 2)                               AS mean_colourfulness,
        round(avg(CASE WHEN has_face THEN 1.0 ELSE 0.0 END), 4)    AS face_second_share,
        round(avg(CASE WHEN has_screen_text THEN 1.0 ELSE 0.0 END), 4) AS text_second_share,
        round(avg(mean_faces), 3)                                  AS mean_faces_on_screen,
        round(max(largest_face_area_frac), 4)                      AS max_face_area_frac,
        mode(modal_face_emotion)                                   AS modal_face_emotion,
        round(avg(CASE WHEN audio_class IN ('speech', 'speech_over_music')
                       THEN 1.0 ELSE 0.0 END), 4)                  AS speech_second_share,
        round(avg(CASE WHEN audio_class = 'music' THEN 1.0 ELSE 0.0 END), 4) AS music_second_share,
        round(avg(CASE WHEN audio_class = 'silence' THEN 1.0 ELSE 0.0 END), 4) AS silence_second_share,
        mode(audio_class)                                          AS modal_audio_class,
        round(max(max_persons), 0)                                 AS max_persons_on_screen
    FROM creative_seconds
    GROUP BY creative_id
),
text_rollup AS (
    SELECT
        creative_id,
        count(*)                                        AS n_text_spans,
        round(sum(duration_s), 2)                       AS total_text_s,
        arg_max(text, max_height_frac)                  AS largest_text
    FROM screen_text
    GROUP BY creative_id
),
object_rollup AS (
    SELECT
        creative_id,
        count(DISTINCT class_name)                      AS n_distinct_object_classes,
        arg_max(class_name, n) FILTER (WHERE class_name <> 'person') AS top_non_person_object
    FROM (
        SELECT creative_id, class_name, count(*) AS n
        FROM raw_object_detections
        GROUP BY creative_id, class_name
    )
    GROUP BY creative_id
)
SELECT
    c.creative_id,
    c.title,
    c.source_label,
    c.display_name,
    c.duration_s,
    c.orientation,
    printf('%dx%d', c.width, c.height)      AS resolution,

    -- pacing
    c.n_shots,
    c.mean_shot_s,
    c.median_shot_s,
    c.cuts_per_minute,

    -- voice
    c.language,
    c.n_spoken_words,
    c.words_per_minute,
    c.speech_share,

    -- sound
    c.tempo_bpm,
    c.mean_dbfs,
    c.dynamic_range_db,
    s.speech_second_share,
    s.music_second_share,
    s.silence_second_share,
    s.modal_audio_class,

    -- look
    c.modal_shot_scale,
    c.modal_setting,
    c.modal_subject,
    c.modal_style,
    s.mean_brightness,
    s.mean_saturation,
    s.mean_colourfulness,
    s.mean_motion,

    -- people
    s.face_second_share,
    s.mean_faces_on_screen,
    s.max_face_area_frac,
    s.modal_face_emotion,
    s.max_persons_on_screen,

    -- text and objects
    s.text_second_share,
    t.n_text_spans,
    t.total_text_s,
    t.largest_text,
    o.n_distinct_object_classes,
    o.top_non_person_object,

    s.n_seconds,
    c.n_sampled_frames,
    c.file_sha256
FROM creatives c
LEFT JOIN second_rollup s USING (creative_id)
LEFT JOIN text_rollup   t USING (creative_id)
LEFT JOIN object_rollup o USING (creative_id)
ORDER BY c.creative_id;
