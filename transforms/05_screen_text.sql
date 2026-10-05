-- ---------------------------------------------------------------------------
-- Layer 2: on-screen text as time spans, not as per-frame detections.
--
-- OCR fires once per sampled frame, so a tagline held for four seconds
-- produces four near-identical rows. An analyst wants "this line appeared at
-- 0:11 and stayed for 4s", so consecutive detections of the same normalised
-- string are collapsed into a span. The per-frame detections remain available
-- in `raw_screen_text_detections` for anyone who needs them.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE screen_text AS
WITH cfg AS (
    -- The OCR sampling period, read from the warehouse rather than hardcoded,
    -- so changing OCR_HZ in config.py keeps these spans correct.
    SELECT creative_id, 1.0 / sample_hz AS step_s
    FROM raw_sampling_config
    WHERE extractor = 'ocr'
),
per_frame AS (
    -- One row per (frame, distinct string): OCR often splits and re-detects
    -- the same line, so collapse within the frame first.
    SELECT
        creative_id,
        t_start_s,
        lower(trim(regexp_replace(text, '\s+', ' ', 'g'))) AS text_norm,
        arg_max(text, score)   AS text,
        max(score)             AS score,
        max(height_frac)       AS height_frac,
        avg(centre_y)          AS centre_y,
        sum(area_frac)         AS area_frac,
        count(*)               AS n_boxes
    FROM raw_screen_text_detections
    GROUP BY creative_id, t_start_s, text_norm
),
flagged AS (
    SELECT
        p.*,
        c.step_s,
        -- A gap of more than two sampling periods means the line left the
        -- screen and came back: that is a new span, not one long one.
        CASE
            WHEN t_start_s - lag(t_start_s) OVER (
                     PARTITION BY p.creative_id, p.text_norm ORDER BY t_start_s
                 ) <= 2 * c.step_s
            THEN 0 ELSE 1
        END AS is_new_span
    FROM per_frame p
    JOIN cfg c USING (creative_id)
),
numbered AS (
    SELECT *,
           sum(is_new_span) OVER (
               PARTITION BY creative_id, text_norm ORDER BY t_start_s
           ) AS span_id
    FROM flagged
)
SELECT
    creative_id,
    row_number() OVER (ORDER BY creative_id, min(t_start_s)) AS screen_text_id,
    arg_max(text, score)                       AS text,
    text_norm,
    min(t_start_s)                             AS start_s,
    round(max(t_start_s) + any_value(step_s), 3) AS end_s,
    round(max(t_start_s) + any_value(step_s) - min(t_start_s), 3) AS duration_s,
    count(*)                                   AS n_frames_present,
    round(max(score), 4)                       AS max_score,
    round(max(height_frac), 4)                 AS max_height_frac,
    round(avg(centre_y), 4)                    AS mean_centre_y,
    -- Rough role of the text, from how big it is and where it sits.
    max(height_frac) >= 0.08                   AS is_large_text,
    CASE
        WHEN avg(centre_y) < 0.33 THEN 'upper_third'
        WHEN avg(centre_y) > 0.66 THEN 'lower_third'
        ELSE 'middle_third'
    END                                        AS vertical_position
FROM numbered
GROUP BY creative_id, text_norm, span_id
ORDER BY creative_id, start_s;
