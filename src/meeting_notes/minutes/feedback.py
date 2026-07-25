"""Parse user corrections from rendered Markdown notes and update summary and glossary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import structlog

from meeting_notes.transcript.glossary import add_term_to_glossary

log = structlog.get_logger()


@dataclass
class ParsedFeedbackItem:
    """A user feedback entry parsed from Markdown notes."""

    checked: bool
    category_label: str
    question: str
    suggested_correction: str | None
    evidence: list[str]
    answer: str


def parse_markdown_feedback(markdown_content: str) -> list[ParsedFeedbackItem]:
    """Parse user responses from the '## 사용자 확인 및 정정' section of Markdown notes.

    Looks for checklist items matching:
      - [x] **[ASR 정정]** question (추천 정정: `suggested`)
        - 근거: [seg-000012](...)
        - 답변: user answer here
    """
    items: list[ParsedFeedbackItem] = []

    # Find the section ## 사용자 확인 및 정정
    section_match = re.search(
        r"^##\s+사용자 확인 및 정정\s*$(.*?)(?=^##|\Z)",
        markdown_content,
        re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return items

    section_text = section_match.group(1)

    # Split into item blocks starting with - [ ] or - [x]
    item_blocks = re.split(r"(?=- \[(?: |x|X)\])", section_text)

    for block in item_blocks:
        block = block.strip()
        if not block.startswith("- ["):
            continue

        # Check state
        checked = block.startswith("- [x]") or block.startswith("- [X]")

        # Extract header line: - [ ] **[CATEGORY]** Question (추천 정정: `suggestion`)
        header_match = re.search(
            r"- \[(?: |x|X)\]\s+\*\*\[(.*?)\]\*\*\s+(.*?)(?:\s+\(추천 정정:\s*`(.*?)`\))?$",
            block.splitlines()[0],
        )
        if not header_match:
            continue

        cat_label = header_match.group(1).strip()
        question = header_match.group(2).strip()
        suggested = header_match.group(3).strip() if header_match.group(3) else None

        # Extract evidence segment IDs if present
        evidence: list[str] = []
        ev_match = re.search(r"-\s+근거:\s+(.*)", block)
        if ev_match:
            evidence = list(dict.fromkeys(re.findall(r"seg-\d+", ev_match.group(1))))

        # Extract user answer line: - 답변: user text
        answer = ""
        ans_match = re.search(r"-\s+답변:\s*(.*)", block)
        if ans_match:
            answer = ans_match.group(1).strip()

        if checked or answer:
            items.append(
                ParsedFeedbackItem(
                    checked=checked,
                    category_label=cat_label,
                    question=question,
                    suggested_correction=suggested,
                    evidence=evidence,
                    answer=answer,
                )
            )

    log.info("feedback.parsed", count=len(items))
    return items


def apply_user_feedback(
    summary: dict,
    feedback_items: list[ParsedFeedbackItem],
    *,
    glossary_path: Path | None = None,
) -> tuple[dict, int]:
    """Apply parsed user feedback items to summary dict and update glossary.

    Returns (updated_summary, applied_count).
    """
    if not feedback_items:
        return summary, 0

    applied_count = 0
    updated_summary = dict(summary)
    user_clarifications = list(updated_summary.get("user_clarifications", []))

    for item in feedback_items:
        # Match feedback item with user_clarifications entry by evidence or question
        matched_clarification = None
        for uc in user_clarifications:
            if uc.get("evidence") and item.evidence and set(uc.get("evidence", [])) & set(item.evidence):
                matched_clarification = uc
                break
            if uc.get("question", "").strip() == item.question.strip():
                matched_clarification = uc
                break

        final_answer = item.answer or item.suggested_correction or ""
        if not final_answer and not item.checked:
            continue

        applied_count += 1

        if matched_clarification:
            matched_clarification["user_answer"] = final_answer
            matched_clarification["resolved"] = True

        # If glossary_path provided and item represents an ASR correction or term clarification
        if glossary_path and final_answer:
            # Extract possible alias from question or suggested_correction
            alias = item.suggested_correction
            if not alias:
                # Look for quoted text in question like "아르고 시디"
                quoted = re.findall(r"[\"']([^\"']+)[\"']", item.question)
                if quoted:
                    alias = quoted[0]

            add_term_to_glossary(glossary_path, canonical=final_answer, alias=alias)

    updated_summary["user_clarifications"] = user_clarifications
    log.info("feedback.applied", count=applied_count)
    return updated_summary, applied_count
