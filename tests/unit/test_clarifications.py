"""Tests for the clarify sidecar workflow: templates, staleness, glossary
scoping, transcript correction, and re-publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meeting_notes.clarifications import (
    ClarificationError,
    apply_clarifications,
    load_answers,
    write_template,
)
from meeting_notes.config import MeetingNotesConfig
from meeting_notes.jobs import load_manifest, save_manifest
from meeting_notes.transcript.glossary import (
    Glossary,
    GlossaryTerm,
    correct_transcript_segments,
    load_glossary,
    load_layered_glossary,
    merge_glossaries,
)


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "2026-07-25-demo"
    (job / "transcript").mkdir(parents=True)
    (job / "summary").mkdir()
    (job / "source").mkdir()
    (job / "source" / "recording.m4a").write_bytes(b"recording")

    transcript = {
        "metadata": {"language": "ko"},
        "segments": [
            {"id": "seg-000012", "start": 12.0, "end": 16.0, "text": "아르고 시디를 배포합니다."},
            {
                "id": "seg-000013",
                "start": 16.0,
                "end": 20.0,
                "text": "아르고 시디 설정을 확인해주세요.",
            },
        ],
    }
    (job / "transcript" / "transcript.merged.json").write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "title": "API Auth Meeting",
        "short_title": "API Auth Review",
        "meeting_date": "2026-07-25",
        "user_clarifications": [
            {
                "category": "asr_correction",
                "question": '"아르고 시디"로 전사됨. "ArgoCD"가 맞나요?',
                "heard_text": "아르고 시디",
                "suggested_correction": "ArgoCD",
                "evidence": ["seg-000012"],
                "user_answer": None,
                "resolved": False,
            },
            {
                "category": "missing_info",
                "question": "담당자가 명시되지 않았습니다. 누구인가요?",
                "heard_text": None,
                "suggested_correction": None,
                "evidence": ["seg-000013"],
                "user_answer": None,
                "resolved": False,
            },
        ],
    }
    (job / "summary" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )

    manifest = load_manifest(job)
    manifest["source"].update(
        {"original_filename": "recording.m4a", "original_path": str(job / "recording.m4a")}
    )
    save_manifest(job, manifest)
    return job


def _config(tmp_path: Path, *, summarization_enabled: bool = True) -> MeetingNotesConfig:
    return MeetingNotesConfig(
        glossary={"path": str(tmp_path / "global-glossary.yaml")},
        summarization={"enabled": summarization_enabled},
    )


def _answer(job: Path, path: Path, answers: dict[str, str]) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item_id, answer in answers.items():
        document["clarifications"][item_id]["answer"] = answer
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")


def _comment(job: Path, path: Path, comments: list[str]) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["comments"] = comments
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")


class TestWriteTemplate:
    def test_builds_one_entry_per_open_clarification(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        path, warning = write_template(job)
        assert warning is None
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["version"] == 1
        assert document["job_id"] == job.name
        assert set(document["clarifications"]) == {"clarif-000", "clarif-001"}
        assert document["clarifications"]["clarif-000"]["heard_text"] == "아르고 시디"
        assert document["clarifications"]["clarif-000"]["answer"] == ""
        assert document["comments"] == [""]

    def test_no_clarifications_returns_none(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        summary_path = job / "summary" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["user_clarifications"] = []
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        path, warning = write_template(job)
        assert path is None
        assert warning is None

    def test_regenerating_preserves_typed_answers(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})

        write_template(job, force=True)
        regenerated = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert regenerated["clarifications"]["clarif-000"]["answer"] == "ArgoCD"
        assert list(job.glob("clarifications.yaml.bak-*"))

    def test_regenerating_preserves_typed_comments(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _comment(job, path, ["Prefer expanding acronyms on first use.", ""])

        write_template(job, force=True)
        regenerated = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert regenerated["comments"] == ["Prefer expanding acronyms on first use."]


class TestLoadAnswers:
    def test_stale_transcript_is_rejected(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["transcript_sha256"] = "stale"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

        with pytest.raises(ClarificationError, match="template --force"):
            load_answers(job, path)

    def test_wrong_job_id_is_rejected(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["job_id"] = "some-other-job"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

        with pytest.raises(ClarificationError, match="belongs to job"):
            load_answers(job, path)


class TestApplyClarifications:
    def _patch_summarizer(self, monkeypatch: pytest.MonkeyPatch, capture: dict) -> None:
        def fake_summarize_transcript(segments, config, local_only, *, extra_context=""):
            capture["extra_context"] = extra_context
            capture["segments"] = segments
            return {
                "title": "API Auth Meeting",
                "short_title": "API Auth Review",
                "meeting_date": "2026-07-25",
                "executive_summary": ["ArgoCD 배포를 논의했습니다."],
                "user_clarifications": [],
            }

        monkeypatch.setattr(
            "meeting_notes.pipeline._summarize_transcript", fake_summarize_transcript
        )

    def test_glossary_direction_and_category_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD", "clarif-001": "김철수"})
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        record = apply_clarifications(job, path, _config(tmp_path))

        assert record["applied_count"] == 2
        assert record["glossary_terms_added"] == 1

        job_glossary = load_glossary(job / "glossary.yaml")
        assert len(job_glossary.terms) == 1
        assert job_glossary.terms[0].canonical == "ArgoCD"
        assert job_glossary.terms[0].aliases == ["아르고 시디"]

        # missing_info answer must not become a glossary term anywhere
        raw = (job / "glossary.yaml").read_text(encoding="utf-8")
        assert "김철수" not in raw

        # global glossary is untouched until an explicit promote
        assert not Path(_config(tmp_path).glossary.path).exists()

    def test_transcript_round_trip_after_glossary_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assertion the original feedback.py lacked: after applying an
        answer, the resulting glossary must actually correct the misheard text."""
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        apply_clarifications(job, path, _config(tmp_path))

        transcript = json.loads(
            (job / "transcript" / "transcript.merged.json").read_text(encoding="utf-8")
        )
        texts = [s["text"] for s in transcript["segments"]]
        assert any("ArgoCD" in t for t in texts)

        job_glossary = load_glossary(job / "glossary.yaml")
        corrected, corrections = correct_transcript_segments(
            [{"id": "x", "text": "아르고 시디 배포 확인"}], job_glossary
        )
        assert "ArgoCD" in corrected[0]["text"]
        assert len(corrections) == 1

    def test_extra_context_reaches_summarizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        apply_clarifications(job, path, _config(tmp_path))

        assert "ArgoCD" in capture["extra_context"]
        assert "human reviewer" in capture["extra_context"]

    def test_resolved_answers_persist_through_resummarize(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        apply_clarifications(job, path, _config(tmp_path))

        summary = json.loads((job / "summary" / "summary.json").read_text(encoding="utf-8"))
        clarification = summary["user_clarifications"][0]
        assert clarification["resolved"] is True
        assert clarification["user_answer"] == "ArgoCD"

    def test_publishes_new_generation_and_supersedes_previous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        first = apply_clarifications(job, path, _config(tmp_path))
        manifest = load_manifest(job)
        assert manifest["clarification_publications"]["active_generation"] == first["id"]
        assert all(Path(item).exists() for item in first["managed_paths"])

        # Second apply requires a fresh template against the corrected transcript.
        path2, _ = write_template(job, force=True)
        _answer(job, path2, {"clarif-001": "김철수"})
        second = apply_clarifications(job, path2, _config(tmp_path))

        manifest = load_manifest(job)
        generations = manifest["clarification_publications"]["generations"]
        states = [g["state"] for g in generations]
        assert states == ["superseded", "active"]
        assert manifest["clarification_publications"]["active_generation"] == second["id"]

    def test_no_answers_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        with pytest.raises(ClarificationError, match="No answers"):
            apply_clarifications(job, path, _config(tmp_path))

    def test_comment_only_run_reaches_summarizer_without_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _comment(job, path, ["When unsure about a name, prefer the romanized spelling."])
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        record = apply_clarifications(job, path, _config(tmp_path))

        assert record["applied_count"] == 0
        assert record["comment_count"] == 1
        assert "romanized spelling" in capture["extra_context"]
        assert "reviewer notes" in capture["extra_context"]
        # no item was answered, so no glossary term should be written
        assert not (job / "glossary.yaml").exists()

    def test_comments_accompany_answers_in_extra_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _answer(job, path, {"clarif-000": "ArgoCD"})
        _comment(job, path, ["Treat all Kubernetes references as ArgoCD-managed."])
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        apply_clarifications(job, path, _config(tmp_path))

        assert "ArgoCD" in capture["extra_context"]
        assert "ArgoCD-managed" in capture["extra_context"]

    def test_whitespace_only_comments_are_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = _job(tmp_path)
        path, _ = write_template(job)
        _comment(job, path, ["", "   "])
        capture: dict = {}
        self._patch_summarizer(monkeypatch, capture)

        with pytest.raises(ClarificationError, match="No answers"):
            apply_clarifications(job, path, _config(tmp_path))


class TestGlossaryLayering:
    def test_merge_unions_aliases_for_shared_canonical(self) -> None:
        base = Glossary(terms=[GlossaryTerm(canonical="ArgoCD", aliases=["아르고 시디"])])
        overlay = Glossary(terms=[GlossaryTerm(canonical="ArgoCD", aliases=["아르고씨디"])])

        merged = merge_glossaries(base, overlay)

        assert len(merged.terms) == 1
        assert sorted(merged.terms[0].aliases) == ["아르고 시디", "아르고씨디"]

    def test_merge_appends_overlay_only_terms(self) -> None:
        base = Glossary(terms=[GlossaryTerm(canonical="ArgoCD", aliases=["아르고 시디"])])
        overlay = Glossary(terms=[GlossaryTerm(canonical="K8s", aliases=["케이 에잇 에스"])])

        merged = merge_glossaries(base, overlay)

        assert [t.canonical for t in merged.terms] == ["ArgoCD", "K8s"]

    def test_load_layered_glossary_does_not_mutate_global_file(self, tmp_path: Path) -> None:
        from meeting_notes.transcript.glossary import save_glossary

        global_path = tmp_path / "global.yaml"
        job_path = tmp_path / "job.yaml"
        save_glossary(
            Glossary(terms=[GlossaryTerm(canonical="ArgoCD", aliases=["아르고 시디"])]), global_path
        )
        save_glossary(
            Glossary(terms=[GlossaryTerm(canonical="K8s", aliases=["케이 에잇 에스"])]), job_path
        )
        before = global_path.read_text(encoding="utf-8")

        layered = load_layered_glossary(global_path, job_path)

        assert {t.canonical for t in layered.terms} == {"ArgoCD", "K8s"}
        assert global_path.read_text(encoding="utf-8") == before

    def test_missing_job_glossary_falls_back_to_global(self, tmp_path: Path) -> None:
        from meeting_notes.transcript.glossary import save_glossary

        global_path = tmp_path / "global.yaml"
        save_glossary(
            Glossary(terms=[GlossaryTerm(canonical="ArgoCD", aliases=["아르고 시디"])]), global_path
        )

        layered = load_layered_glossary(global_path, tmp_path / "missing.yaml")

        assert [t.canonical for t in layered.terms] == ["ArgoCD"]
