from __future__ import annotations

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
    # Extra AGENTS.md guardrails. The requested exact phrases above remain
    # first-class test targets; these keep generated text cautious.
    "破局寸前",
    "依存している",
    "病的",
    "診断できます",
)

FORBIDDEN_REPLACEMENTS = {
    "確実に喧嘩していた": "喧嘩候補がありました",
    "相手は冷めていた": "相手の気持ちは断定できません",
    "相手は怒っていた": "相手の気持ちは断定できません",
    "相手は愛情がなかった": "相手の気持ちは断定できません",
    "あなたが悪い": "責任の所在は記録から断定しません",
    "相手が悪い": "責任の所在は記録から断定しません",
    "関係は終わっていた": "関係状態は記録から断定しません",
    "この人は恋人です": "関係ラベルは手動設定が必要です",
    "この人は親密な関係です": "親密度は推定しません",
    "この写真はデート確定です": "この写真は外出候補です",
    "破局寸前": "強い断定は避けます",
    "依存している": "心理状態は診断しません",
    "病的": "心理状態は診断しません",
    "診断できます": "診断はできません",
}

PUBLIC_RELATIONSHIP_LABELS = (
    "恋人",
    "恋人ラベル",
    "彼女",
    "彼氏",
    "元恋人",
    "partner",
    "ex_partner",
    "close_person",
    "other_private",
    "lover",
    "girlfriend",
    "boyfriend",
    "ex-partner",
)

PUBLIC_REDACTION_MARKERS = (
    "[LINE full text redacted]",
    "[note full text redacted]",
    "[exact GPS redacted]",
    "[face data redacted]",
    "[real photo redacted]",
    "[private path redacted]",
    "[source file path redacted]",
    "[relationship label redacted]",
)

INNER_FEELING_REPLACEMENT = "相手の内心は記録から断定できません"

PUBLIC_ANONYMOUS_PERSON_LABELS = tuple(f"人物{chr(ord('A') + index)}" for index in range(26))
