PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS relationship_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_name TEXT NOT NULL,
  person_source_id TEXT,
  line_speaker_source_id TEXT,
  line_speaker_group_source_id TEXT,
  self_person_source_id TEXT,
  self_line_speaker_source_id TEXT,
  self_line_speaker_group_source_id TEXT,
  relationship_label TEXT,
  label_source TEXT NOT NULL DEFAULT 'user_manual',
  valid_from TEXT,
  valid_to TEXT,
  visibility TEXT NOT NULL DEFAULT 'private',
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (relationship_label IS NULL OR relationship_label IN ('partner', 'ex_partner', 'close_person', 'other_private')),
  CHECK (label_source = 'user_manual'),
  CHECK (visibility IN ('private', 'hidden'))
);

CREATE TABLE IF NOT EXISTS relationship_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER,
  event_type TEXT NOT NULL,
  event_date TEXT NOT NULL,
  summary TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'candidate',
  review_status TEXT NOT NULL DEFAULT 'unreviewed',
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_strength REAL NOT NULL DEFAULT 0.5,
  severity INTEGER NOT NULL DEFAULT 0,
  generated_by_model TEXT,
  prompt_version TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (profile_id) REFERENCES relationship_profiles(id) ON DELETE SET NULL,
  CHECK (event_type IN (
    'conflict',
    'minor_misunderstanding',
    'reconciliation',
    'promise',
    'date_or_outing',
    'reply_delay',
    'call_event',
    'emotional_note',
    'affection_note',
    'important_memory',
    'anniversary',
    'unresolved_issue',
    'other'
  )),
  CHECK (status IN ('candidate', 'hidden', 'archived')),
  CHECK (review_status IN ('unreviewed', 'verified', 'corrected', 'needs_reanalysis', 'rejected')),
  CHECK (confidence >= 0.0 AND confidence <= 1.0),
  CHECK (evidence_strength >= 0.0 AND evidence_strength <= 1.0),
  CHECK (severity >= 0 AND severity <= 4)
);

CREATE TABLE IF NOT EXISTS relationship_event_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER NOT NULL,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_pointer TEXT,
  source_date TEXT,
  role TEXT NOT NULL DEFAULT 'supporting',
  summary TEXT NOT NULL,
  excerpt TEXT,
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_strength REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (event_id) REFERENCES relationship_events(id) ON DELETE CASCADE,
  CHECK (source_type IN (
    'line_message',
    'line_call',
    'photo',
    'video',
    'gps',
    'place',
    'media_caption',
    'ocr',
    'note',
    'thought',
    'note_event',
    'monthly_reflection',
    'manual'
  )),
  CHECK (role IN ('primary', 'supporting', 'contradictory', 'context', 'before_event', 'after_event')),
  CHECK (confidence >= 0.0 AND confidence <= 1.0),
  CHECK (evidence_strength >= 0.0 AND evidence_strength <= 1.0)
);

CREATE TABLE IF NOT EXISTS interaction_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER,
  metric_date TEXT NOT NULL,
  metric_type TEXT NOT NULL,
  metric_value REAL NOT NULL,
  source_pointer TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (profile_id) REFERENCES relationship_profiles(id) ON DELETE SET NULL,
  CHECK (metric_type IN (
    'message_count',
    'reply_delay',
    'longest_silence',
    'call_count',
    'call_duration',
    'apology_signal_count',
    'promise_signal_count',
    'conflict_signal_count'
  ))
);

CREATE TABLE IF NOT EXISTS post_conflict_activities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conflict_event_id INTEGER,
  activity_event_id INTEGER,
  activity_date TEXT NOT NULL,
  days_after_conflict INTEGER NOT NULL,
  place_label TEXT,
  activity_type TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  evidence_strength REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (conflict_event_id) REFERENCES relationship_events(id) ON DELETE SET NULL,
  FOREIGN KEY (activity_event_id) REFERENCES relationship_events(id) ON DELETE SET NULL,
  CHECK (confidence >= 0.0 AND confidence <= 1.0),
  CHECK (evidence_strength >= 0.0 AND evidence_strength <= 1.0)
);

CREATE TABLE IF NOT EXISTS relationship_review_actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id INTEGER,
  action TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (event_id) REFERENCES relationship_events(id) ON DELETE SET NULL,
  CHECK (action IN (
    'verify',
    'reject',
    'correct',
    'hide',
    'pin',
    'mark_as_joke',
    'mark_as_misunderstanding',
    'mark_as_reconciled',
    'mark_as_unresolved',
    'needs_reanalysis',
    'dismiss_warning',
    'revert'
  ))
);

CREATE TABLE IF NOT EXISTS llm_analysis_cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_type TEXT NOT NULL,
  source_window_hash TEXT NOT NULL,
  model_name TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  result_json TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (analysis_type, source_window_hash, model_name, prompt_version),
  CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TRIGGER IF NOT EXISTS relationship_profiles_set_updated_at
AFTER UPDATE ON relationship_profiles
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE relationship_profiles SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS relationship_events_set_updated_at
AFTER UPDATE ON relationship_events
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE relationship_events SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS relationship_event_evidence_set_updated_at
AFTER UPDATE ON relationship_event_evidence
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE relationship_event_evidence SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS interaction_metrics_set_updated_at
AFTER UPDATE ON interaction_metrics
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE interaction_metrics SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS post_conflict_activities_set_updated_at
AFTER UPDATE ON post_conflict_activities
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE post_conflict_activities SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS relationship_review_actions_set_updated_at
AFTER UPDATE ON relationship_review_actions
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE relationship_review_actions SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS llm_analysis_cache_set_updated_at
AFTER UPDATE ON llm_analysis_cache
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE llm_analysis_cache SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;
