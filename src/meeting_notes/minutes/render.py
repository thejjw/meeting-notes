"""Deterministic meeting minutes rendering from summary JSON."""

from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()


def render_minutes(
    summary: dict,
    *,
    source_filename: str = "",
    duration_timestamp: str = "",
    title_template: str = "{date} {title}",
    include_executive_summary: bool = True,
    include_participants: bool = True,
    include_agenda: bool = True,
    include_decisions: bool = True,
    include_action_items: bool = True,
    include_open_questions: bool = True,
    include_risks: bool = True,
    include_user_clarifications: bool = True,
    include_evidence_links: bool = True,
    include_full_transcript_link: bool = True,
    unknown_value_text: str = "미정",
) -> str:
    """Render meeting minutes as Markdown from a summary dict.

    This is a pure local renderer — no LLM involved.
    """
    title = summary.get("title", "Meeting Notes")
    date = summary.get("meeting_date") or unknown_value_text
    short_title = summary.get("short_title", "")

    # Title
    doc_title = title_template.format(date=date, title=title)
    lines = [f"# {doc_title}", ""]

    # Metadata
    if source_filename:
        lines.append(f"- 원본 녹음: `{source_filename}`")
    if duration_timestamp:
        lines.append(f"- 길이: {duration_timestamp}")
    if date and date != unknown_value_text:
        lines.append(f"- 일시: {date}")

    # Participants
    if include_participants and summary.get("participants"):
        participants = summary["participants"]
        names = [p.get("name", unknown_value_text) for p in participants]
        lines.append(f"- 참석자: {', '.join(names)}")

    lines.append("")

    # Executive summary
    if include_executive_summary and summary.get("executive_summary"):
        lines.append("## 핵심 요약")
        lines.append("")
        for point in summary["executive_summary"]:
            lines.append(f"- {point}")
        lines.append("")

    # Agenda
    if include_agenda and summary.get("agenda"):
        lines.append("## 논의 안건")
        lines.append("")
        for item in summary["agenda"]:
            lines.append(f"- {item}")
        lines.append("")

    # Discussion topics (chronological)
    if summary.get("discussion_topics"):
        lines.append("## 논의 사항")
        lines.append("")
        for topic in summary["discussion_topics"]:
            topic_name = topic.get("topic", "")
            time_range = topic.get("time_range", "")
            time_label = f" ({time_range})" if time_range else ""
            lines.append(f"### {topic_name}{time_label}")
            lines.append("")
            for point in topic.get("summary", []):
                lines.append(f"- {point}")
            if include_evidence_links and topic.get("evidence"):
                evidence = ", ".join(f"[{e}](../transcript/transcript.merged.md#{e})" for e in topic["evidence"])
                lines.append(f"\n- 근거: {evidence}")
            lines.append("")

    # Decisions
    if include_decisions and summary.get("decisions"):
        lines.append("## 확정 사항")
        lines.append("")
        for i, dec in enumerate(summary["decisions"], 1):
            status = dec.get("status", "")
            status_label = f" ({status})" if status else ""
            lines.append(f"{i}. {dec.get('decision', '')}{status_label}")
            if include_evidence_links and dec.get("evidence"):
                evidence = ", ".join(f"[{e}](../transcript/transcript.merged.md#{e})" for e in dec["evidence"])
                lines.append(f"   - 근거: {evidence}")
        lines.append("")

    # Action items
    if include_action_items and summary.get("action_items"):
        lines.append("## 후속 조치")
        lines.append("")
        lines.append("| 담당자 | 작업 | 기한 | 근거 |")
        lines.append("|---|---|---|---|")
        for item in summary["action_items"]:
            owner = item.get("owner") or unknown_value_text
            due = item.get("due_date") or unknown_value_text
            task = item.get("task", "")
            evidence = ""
            if include_evidence_links and item.get("evidence"):
                evidence = ", ".join(f"[{e}](../transcript/transcript.merged.md#{e})" for e in item["evidence"])
            lines.append(f"| {owner} | {task} | {due} | {evidence} |")
        lines.append("")

    # Open questions
    if include_open_questions and summary.get("open_questions"):
        lines.append("## 미해결 질문")
        lines.append("")
        for q in summary["open_questions"]:
            lines.append(f"- {q.get('question', '')}")
        lines.append("")

    # Risks
    if include_risks and summary.get("risks"):
        lines.append("## 위험 및 고려사항")
        lines.append("")
        for risk in summary["risks"]:
            risk_text = risk.get("risk", "")
            impact = risk.get("impact")
            mitigation = risk.get("mitigation")
            lines.append(f"- {risk_text}")
            if impact:
                lines.append(f"  - 영향: {impact}")
            if mitigation:
                lines.append(f"  - 대응방법: {mitigation}")
        lines.append("")

    # User clarifications & ASR corrections
    if include_user_clarifications and summary.get("user_clarifications"):
        lines.append("## 사용자 확인 및 정정")
        lines.append("")
        category_labels = {
            "asr_correction": "ASR 정정",
            "missing_info": "정보 확인",
            "term_clarification": "용어 확인",
        }
        for item in summary["user_clarifications"]:
            cat = category_labels.get(item.get("category", ""), "확인 요청")
            question = item.get("question", "")
            suggestion = item.get("suggested_correction")
            sugg_text = f" (추천 정정: `{suggestion}`)" if suggestion else ""
            lines.append(f"- [ ] **[{cat}]** {question}{sugg_text}")
            if include_evidence_links and item.get("evidence"):
                evidence = ", ".join(f"[{e}](../transcript/transcript.merged.md#{e})" for e in item["evidence"])
                lines.append(f"  - 근거: {evidence}")
            lines.append("  - 답변: ")
        lines.append("")

    # Transcription uncertainties
    if summary.get("transcription_uncertainties"):
        lines.append("## 전사 불확실 구간")
        lines.append("")
        for u in summary["transcription_uncertainties"]:
            lines.append(f"- {u.get('description', '')}")
        lines.append("")

    return "\n".join(lines)


def save_minutes(markdown: str, output_path: Path) -> Path:
    """Save meeting minutes to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
