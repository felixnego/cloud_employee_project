-- ---------------------------------------------------------------------------
-- Layer 3: materialise the remaining detail grains as real tables.
--
-- Why not leave them as views over Parquet? Because `warehouse.duckdb` should
-- be self-contained. The raw_* views hold absolute paths, so a database file
-- emailed to a colleague would have working marts and broken views. Anything
-- an analyst is expected to query is therefore a table.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE face_detections AS
SELECT f.*, round(f.t_start_s / nullif(c.duration_s, 0), 4) AS position_in_ad
FROM raw_face_detections f
JOIN creatives c USING (creative_id)
ORDER BY f.creative_id, f.frame_index, f.face_index;

CREATE OR REPLACE TABLE object_detections AS
SELECT o.*, round(o.t_start_s / nullif(c.duration_s, 0), 4) AS position_in_ad
FROM raw_object_detections o
JOIN creatives c USING (creative_id)
ORDER BY o.creative_id, o.frame_index, o.detection_index;

-- Long form, all labels and scores -- not just the winner. Keeps "how
-- confident was CLIP that this was a studio rather than indoors?" answerable.
CREATE OR REPLACE TABLE shot_label_scores AS
SELECT * FROM raw_shot_label_scores
ORDER BY creative_id, shot_index, axis, rank;

CREATE OR REPLACE TABLE screen_text_detections AS
SELECT * FROM raw_screen_text_detections
ORDER BY creative_id, frame_index, detection_index;

-- Provenance travels with the data: which extractor version wrote which rows,
-- how long it took, and whether it failed.
CREATE OR REPLACE TABLE run_manifest AS
SELECT * FROM raw_run_manifest
ORDER BY creative_id, extractor;

CREATE OR REPLACE TABLE sampling_config AS
SELECT * FROM raw_sampling_config
ORDER BY creative_id, extractor;
