-- Run this SQL in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS recent_detections (
    id BIGSERIAL PRIMARY KEY,
    class_name TEXT NOT NULL,
    confidence NUMERIC NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE recent_detections ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public inserts" ON recent_detections;
DROP POLICY IF EXISTS "Allow public reads" ON recent_detections;
DROP POLICY IF EXISTS "Allow public deletes" ON recent_detections;

CREATE POLICY "Allow public inserts" ON recent_detections
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "Allow public reads" ON recent_detections
    FOR SELECT TO anon
    USING (true);

CREATE POLICY "Allow public deletes" ON recent_detections
    FOR DELETE TO anon
    USING (true);
