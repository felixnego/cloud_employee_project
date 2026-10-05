-- ---------------------------------------------------------------------------
-- Layer 1: views over the raw Parquet written by the extractors.
--
-- Everything downstream reads these views, never the files, so the physical
-- layout of warehouse/raw/ can change without touching a single mart.
-- `union_by_name` means a new column added by one extractor version does not
-- break the read of files written by the previous one.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW raw_creative_metadata AS
    SELECT * FROM read_parquet('${RAW}/creative_metadata/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_sampling_config AS
    SELECT * FROM read_parquet('${RAW}/sampling_config/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_shot_boundaries AS
    SELECT * FROM read_parquet('${RAW}/shot_boundaries/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_transcript_segments AS
    SELECT * FROM read_parquet('${RAW}/transcript_segments/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_transcript_words AS
    SELECT * FROM read_parquet('${RAW}/transcript_words/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_transcript_summary AS
    SELECT * FROM read_parquet('${RAW}/transcript_summary/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_visual_frames AS
    SELECT * FROM read_parquet('${RAW}/visual_frames/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_audio_frames AS
    SELECT * FROM read_parquet('${RAW}/audio_frames/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_audio_summary AS
    SELECT * FROM read_parquet('${RAW}/audio_summary/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_screen_text_detections AS
    SELECT * FROM read_parquet('${RAW}/screen_text_detections/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_face_detections AS
    SELECT * FROM read_parquet('${RAW}/face_detections/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_object_detections AS
    SELECT * FROM read_parquet('${RAW}/object_detections/*.parquet', union_by_name = true);

CREATE OR REPLACE VIEW raw_shot_label_scores AS
    SELECT * FROM read_parquet('${RAW}/shot_label_scores/*.parquet', union_by_name = true);

-- Provenance: which extractor version produced which rows, when, and how long
-- it took. Queryable so a stale or failed extraction is visible in SQL.
CREATE OR REPLACE VIEW raw_run_manifest AS
    SELECT * FROM read_parquet('${RAW}/_manifest/*.parquet', union_by_name = true);
