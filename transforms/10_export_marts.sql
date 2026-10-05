-- ---------------------------------------------------------------------------
-- Layer 4: publish every mart as Parquet next to the database.
--
-- The DuckDB file is the convenient interface; these files are the portable
-- one. Pandas, Polars, Spark, BigQuery, Snowflake and MotherDuck all read
-- them without this repo, this image, or Python. That is the integration
-- surface offered to engineering stakeholders.
-- ---------------------------------------------------------------------------

COPY creatives              TO '${MARTS}/creatives.parquet'              (FORMAT PARQUET, COMPRESSION ZSTD);
COPY creative_summary       TO '${MARTS}/creative_summary.parquet'       (FORMAT PARQUET, COMPRESSION ZSTD);
COPY shots                  TO '${MARTS}/shots.parquet'                  (FORMAT PARQUET, COMPRESSION ZSTD);
COPY creative_seconds       TO '${MARTS}/creative_seconds.parquet'       (FORMAT PARQUET, COMPRESSION ZSTD);
COPY creative_frames        TO '${MARTS}/creative_frames.parquet'        (FORMAT PARQUET, COMPRESSION ZSTD);
COPY transcript_segments    TO '${MARTS}/transcript_segments.parquet'    (FORMAT PARQUET, COMPRESSION ZSTD);
COPY transcript_words       TO '${MARTS}/transcript_words.parquet'       (FORMAT PARQUET, COMPRESSION ZSTD);
COPY screen_text            TO '${MARTS}/screen_text.parquet'            (FORMAT PARQUET, COMPRESSION ZSTD);
COPY screen_text_detections TO '${MARTS}/screen_text_detections.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
COPY face_detections        TO '${MARTS}/face_detections.parquet'        (FORMAT PARQUET, COMPRESSION ZSTD);
COPY object_detections      TO '${MARTS}/object_detections.parquet'      (FORMAT PARQUET, COMPRESSION ZSTD);
COPY shot_label_scores      TO '${MARTS}/shot_label_scores.parquet'      (FORMAT PARQUET, COMPRESSION ZSTD);
COPY run_manifest           TO '${MARTS}/run_manifest.parquet'           (FORMAT PARQUET, COMPRESSION ZSTD);
COPY sampling_config        TO '${MARTS}/sampling_config.parquet'        (FORMAT PARQUET, COMPRESSION ZSTD);

-- Two CSVs as well: the per-creative and per-shot comparisons are small enough
-- to open in a spreadsheet, which is sometimes the fastest way to look.
COPY creative_summary TO '${MARTS}/creative_summary.csv' (FORMAT CSV, HEADER);
COPY shots            TO '${MARTS}/shots.csv'            (FORMAT CSV, HEADER);
