-- ---------------------------------------------------------------------------
-- Layer 2: the script. Kept at both grains on purpose -- segments read like a
-- script, words are what you need to time a brand mention to the half-second.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE transcript_segments AS
SELECT
    s.creative_id,
    s.segment_index,
    s.start_s,
    s.end_s,
    round(s.end_s - s.start_s, 3)                     AS duration_s,
    s.text,
    s.n_words,
    s.words_per_s,
    round(s.start_s / nullif(c.duration_s, 0), 4)     AS position_in_ad,
    -- Whisper's own confidence signals, exposed rather than hidden: low
    -- avg_logprob or high no_speech_prob means treat the line with suspicion.
    s.avg_logprob,
    s.no_speech_prob,
    s.compression_ratio,
    (s.avg_logprob < -1.0 OR s.no_speech_prob > 0.5)  AS is_low_confidence
FROM raw_transcript_segments s
JOIN creatives c USING (creative_id)
ORDER BY s.creative_id, s.segment_index;

CREATE OR REPLACE TABLE transcript_words AS
SELECT
    w.creative_id,
    w.segment_index,
    w.word_index,
    w.word,
    lower(regexp_replace(w.word, '[^\w'']', '', 'g')) AS word_norm,
    w.start_s,
    w.end_s,
    w.probability,
    round(w.start_s / nullif(c.duration_s, 0), 4)      AS position_in_ad
FROM raw_transcript_words w
JOIN creatives c USING (creative_id)
ORDER BY w.creative_id, w.segment_index, w.word_index;
