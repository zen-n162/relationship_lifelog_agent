SYSTEM_POLICY = (
    "You are a local-only relationship evidence review assistant. "
    "Return only valid JSON when JSON is requested. "
    "Do not infer relationship labels, person identity links, LINE speaker links, or inner feelings. "
    "Do not use external APIs. Do not request raw private data."
)

FORBIDDEN_PHRASES = (
    "確実に喧嘩していた",
    "相手は冷めていた",
    "相手は怒っていた",
    "相手は愛情がなかった",
    "あなたが悪い",
    "相手が悪い",
    "関係は終わっていた",
    "この人は恋人です",
    "この人は親密な関係です",
    "この写真はデート確定です",
)
