"""Glossary-based terminology preservation and correction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
import structlog

log = structlog.get_logger()


@dataclass
class GlossaryTerm:
    """A canonical term with its aliases."""

    canonical: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class Glossary:
    """Complete glossary of terms and aliases."""

    terms: list[GlossaryTerm] = field(default_factory=list)

    @property
    def all_canonical(self) -> list[str]:
        return [t.canonical for t in self.terms]


@dataclass
class GlossaryCorrection:
    """Record of a glossary correction applied to transcript text."""

    segment_id: str
    original_text: str
    corrected_text: str
    rule_canonical: str
    rule_alias: str


def load_glossary(path: Path | None) -> Glossary:
    """Load a glossary from a YAML file.

    Args:
        path: Path to glossary YAML file. None or missing = empty glossary.

    Returns:
        Glossary with loaded terms.
    """
    if not path or not path.exists():
        return Glossary()

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data or "terms" not in data:
        return Glossary()

    terms = []
    for entry in data["terms"]:
        terms.append(
            GlossaryTerm(
                canonical=entry["canonical"],
                aliases=entry.get("aliases", []),
            )
        )

    log.info("glossary.loaded", path=str(path), terms=len(terms))
    return Glossary(terms=terms)


def build_initial_prompt(glossary: Glossary, language: str = "ko") -> str:
    """Build an ASR initial prompt from glossary canonical terms.

    The prompt encourages Whisper to use the correct spellings.
    """
    if not glossary.terms:
        return ""

    # Use canonical terms as prompt hints
    terms_str = ", ".join(glossary.all_canonical)
    return f"Technical terms: {terms_str}"


def apply_glossary_corrections(
    text: str,
    glossary: Glossary,
    *,
    case_sensitive: bool = False,
) -> tuple[str, list[GlossaryCorrection]]:
    """Apply conservative glossary-based text corrections.

    Replaces only whole-word alias matches with canonical forms.
    Never replaces substrings inside unrelated words.

    Args:
        text: Original transcript text.
        glossary: Glossary with terms and aliases.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Tuple of (corrected_text, list_of_corrections).
    """
    if not glossary.terms:
        return text, []

    corrections: list[GlossaryCorrection] = []
    corrected = text

    for term in glossary.terms:
        for alias in term.aliases:
            if not alias:
                continue

            # Build word-boundary regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            # Escape alias for regex, then wrap with word boundaries
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", flags)

            matches = pattern.findall(corrected)
            if matches:
                for match in matches:
                    corrections.append(
                        GlossaryCorrection(
                            segment_id="",
                            original_text=match,
                            corrected_text=term.canonical,
                            rule_canonical=term.canonical,
                            rule_alias=match,
                        )
                    )
                corrected = pattern.sub(term.canonical, corrected)

    return corrected, corrections


def correct_transcript_segments(
    segments: list[dict],
    glossary: Glossary,
    *,
    case_sensitive: bool = False,
) -> tuple[list[dict], list[GlossaryCorrection]]:
    """Apply glossary corrections to a list of transcript segments.

    Only modifies the corrected/reviewed transcript, never raw.json.

    Args:
        segments: List of segment dicts with 'id' and 'text' fields.
        glossary: Glossary to apply.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        Tuple of (corrected_segments, all_corrections).
    """
    all_corrections: list[GlossaryCorrection] = []
    corrected_segments = []

    for seg in segments:
        corrected_text, corrections = apply_glossary_corrections(
            seg.get("text", ""),
            glossary,
            case_sensitive=case_sensitive,
        )

        # Fill in segment IDs for corrections
        for c in corrections:
            c.segment_id = seg.get("id", "")

        all_corrections.extend(corrections)

        new_seg = dict(seg)
        new_seg["text"] = corrected_text
        corrected_segments.append(new_seg)

    if all_corrections:
        log.info(
            "glossary.corrections_applied",
            count=len(all_corrections),
        )

    return corrected_segments, all_corrections
