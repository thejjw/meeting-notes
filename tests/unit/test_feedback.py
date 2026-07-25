"""Tests for user feedback parser, glossary update, and pipeline feedback processing."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.minutes.feedback import apply_user_feedback, parse_markdown_feedback
from meeting_notes.transcript.glossary import load_glossary


class TestParseMarkdownFeedback:
    """Test parsing user feedback from rendered Markdown notes."""

    def test_parse_checked_and_answered_items(self) -> None:
        md = """# Meeting Notes

## 사용자 확인 및 정정

- [x] **[ASR 정정]** "아르고 시디"로 전사됨. "ArgoCD"가 맞나요? (추천 정정: `ArgoCD`)
  - 근거: [seg-000012](../transcript/transcript.merged.md#seg-000012)
  - 답변: ArgoCD
- [ ] **[정보 확인]** "OAuth 전환 작업" 담당자가 미정입니다.
  - 근거: [seg-000045](../transcript/transcript.merged.md#seg-000045)
  - 답변: 김철수
"""
        items = parse_markdown_feedback(md)
        assert len(items) == 2
        assert items[0].checked is True
        assert items[0].category_label == "ASR 정정"
        assert items[0].suggested_correction == "ArgoCD"
        assert items[0].evidence == ["seg-000012"]
        assert items[0].answer == "ArgoCD"

        assert items[1].checked is False
        assert items[1].category_label == "정보 확인"
        assert items[1].evidence == ["seg-000045"]
        assert items[1].answer == "김철수"

    def test_parse_empty_section(self) -> None:
        items = parse_markdown_feedback("# Meeting Notes\n\nNo feedback section here.")
        assert items == []


class TestApplyUserFeedback:
    """Test applying feedback items to summary dict and updating glossary."""

    def test_apply_feedback_updates_summary_and_glossary(self, tmp_path: Path) -> None:
        glossary_file = tmp_path / "glossary.yaml"
        summary = {
            "title": "API Meeting",
            "user_clarifications": [
                {
                    "category": "asr_correction",
                    "question": "Is 'ArgoCD' correct for '아르고 시디'?",
                    "suggested_correction": "ArgoCD",
                    "evidence": ["seg-000012"],
                }
            ],
        }

        md = """## 사용자 확인 및 정정

- [x] **[ASR 정정]** Is 'ArgoCD' correct for '아르고 시디'? (추천 정정: `ArgoCD`)
  - 근거: [seg-000012](../transcript/transcript.merged.md#seg-000012)
  - 답변: ArgoCD
"""
        items = parse_markdown_feedback(md)
        updated, count = apply_user_feedback(summary, items, glossary_path=glossary_file)

        assert count == 1
        assert updated["user_clarifications"][0]["resolved"] is True
        assert updated["user_clarifications"][0]["user_answer"] == "ArgoCD"

        # Verify glossary was updated
        glossary = load_glossary(glossary_file)
        assert len(glossary.terms) == 1
        assert glossary.terms[0].canonical == "ArgoCD"
