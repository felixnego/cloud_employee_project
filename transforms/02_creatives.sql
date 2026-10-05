-- ---------------------------------------------------------------------------
-- Layer 2: the creative dimension. One row per ad, with the attributes that
-- never vary over its runtime, plus whole-ad rollups that are cheap here.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE creatives AS
WITH shot_stats AS (
    SELECT
        creative_id,
        count(*)                         AS n_shots,
        round(avg(duration_s), 3)        AS mean_shot_s,
        round(median(duration_s), 3)     AS median_shot_s,
        round(min(duration_s), 3)        AS min_shot_s,
        round(max(duration_s), 3)        AS max_shot_s
    FROM raw_shot_boundaries
    GROUP BY creative_id
),
label_stats AS (
    -- The ad's modal look, taken from the top CLIP label of each shot.
    SELECT
        creative_id,
        mode(label) FILTER (WHERE axis = 'shot_scale') AS modal_shot_scale,
        mode(label) FILTER (WHERE axis = 'setting')    AS modal_setting,
        mode(label) FILTER (WHERE axis = 'subject')    AS modal_subject,
        mode(label) FILTER (WHERE axis = 'style')      AS modal_style
    FROM raw_shot_label_scores
    WHERE is_top
    GROUP BY creative_id
)
SELECT
    m.creative_id,
    m.title,
    m.source_label,
    -- Titles are not unique (two of the sample creatives are the same campaign
    -- by different creators). `display_name` is the safe label for any chart,
    -- pivot or GROUP BY that a human will read.
    CASE WHEN m.source_label IS NULL OR m.source_label = ''
         THEN m.title
         ELSE m.title || ' - ' || m.source_label
    END                                       AS display_name,
    m.file_name,
    m.duration_s,
    m.width,
    m.height,
    m.orientation,
    m.aspect_ratio,
    m.fps,
    m.video_codec,
    m.has_audio,
    m.audio_codec,
    m.file_size_bytes,
    m.file_sha256,

    -- pacing
    s.n_shots,
    s.mean_shot_s,
    s.median_shot_s,
    s.min_shot_s,
    s.max_shot_s,
    round(60.0 * (s.n_shots - 1) / nullif(m.duration_s, 0), 2) AS cuts_per_minute,

    -- voice
    t.language,
    t.language_probability,
    t.n_segments      AS n_transcript_segments,
    t.n_words         AS n_spoken_words,
    t.speech_s,
    t.speech_share,
    t.words_per_minute,

    -- sound
    a.tempo_bpm,
    a.mean_dbfs,
    a.peak_dbfs,
    a.dynamic_range_db,
    a.silence_share,
    a.mean_harmonic_ratio,

    -- look
    l.modal_shot_scale,
    l.modal_setting,
    l.modal_subject,
    l.modal_style,

    m.sampled_frame_hz,
    m.n_sampled_frames
FROM raw_creative_metadata m
LEFT JOIN shot_stats  s USING (creative_id)
LEFT JOIN label_stats l USING (creative_id)
LEFT JOIN raw_transcript_summary t USING (creative_id)
LEFT JOIN raw_audio_summary      a USING (creative_id)
ORDER BY m.creative_id;
