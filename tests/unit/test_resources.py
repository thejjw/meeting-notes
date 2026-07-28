"""Tests for resource catalog and system probes."""

from __future__ import annotations

from meeting_notes.resources import (
    WHISPER_CPP_RESOURCES,
    check_model_fit,
    detect_system,
    format_diagnostics_table,
    get_resource_estimate,
)


class TestResourceCatalog:
    """Test resource catalog parsing and lookup."""

    def test_whisper_cpp_resources_populated(self) -> None:
        assert len(WHISPER_CPP_RESOURCES) >= 5
        for name in ["tiny", "base", "small", "medium", "large-v3"]:
            assert name in WHISPER_CPP_RESOURCES

    def test_resource_estimate_fields(self) -> None:
        est = WHISPER_CPP_RESOURCES["medium"]
        assert est.model_name == "medium"
        assert est.backend == "whisper_cpp"
        assert est.disk_size_mib > 0
        assert est.reference_memory_mb > 0
        assert est.recommended_free_ram_gb > 0
        assert est.confidence in ("official_reference", "estimated", "measured_local")

    def test_get_resource_estimate(self) -> None:
        est = get_resource_estimate("medium", "whisper_cpp")
        assert est is not None
        assert est.model_name == "medium"

    def test_get_resource_estimate_unknown(self) -> None:
        est = get_resource_estimate("nonexistent", "whisper_cpp")
        assert est is None


class TestModelFit:
    """Test model fit checking against detected system."""

    def test_fit_with_sufficient_ram(self) -> None:
        from meeting_notes.resources import MemoryDetection, SystemDiagnostics

        diag = SystemDiagnostics()
        diag.memory = MemoryDetection(total_ram_gb=128.0, available_ram_gb=100.0)

        est = get_resource_estimate("tiny", "whisper_cpp")
        assert est is not None
        status, reason = check_model_fit(est, diag)
        assert status == "available"

    def test_fit_with_insufficient_ram(self) -> None:
        from meeting_notes.resources import MemoryDetection, SystemDiagnostics

        diag = SystemDiagnostics()
        diag.memory = MemoryDetection(total_ram_gb=2.0, available_ram_gb=1.0)

        est = get_resource_estimate("large-v3", "whisper_cpp")
        assert est is not None
        status, reason = check_model_fit(est, diag)
        assert status in ("not_detected", "available_with_warning")


class TestSystemDetection:
    """Test system detection (may vary by platform)."""

    def test_detect_system_returns_diagnostics(self) -> None:
        diag = detect_system()
        assert diag.os_name
        assert diag.architecture
        assert diag.python_version
        assert diag.cpu.physical_cores >= 1
        assert diag.cpu.logical_cores >= 1
        assert diag.memory.total_ram_gb > 0

    def test_format_diagnostics_table(self) -> None:
        diag = detect_system()
        table = format_diagnostics_table(diag)
        assert "Detected system" in table
        assert "OS:" in table
        assert "CPU:" in table
        assert "System RAM:" in table
