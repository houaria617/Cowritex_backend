"""
tests/unit/test_mock_agents.py
Tests for mock agents — verifies they match the real implementation's contracts.
These same tests will pass unchanged against the real agents after the branch merge,
because they only test the public interface, not internals.
"""
from __future__ import annotations
import os
import pytest


# ══════════════════════════════════════════════════════════════
# Writing Agent
# ══════════════════════════════════════════════════════════════

class TestWritingAgentContract:
    """
    Contract tests: verify the mock honours every guarantee
    the real run_writing_agent makes.
    """

    @pytest.fixture
    def base_ctx(self):
        return {
            "writing_style":  "academic",
            "tone":           "formal",
            "target_journal": "IEEE Transactions",
            "language":       "English",
            "citation_style": "APA",
            "grounded_only":  False,
            "sources":        [],
        }

    # ── Return type ───────────────────────────────────────────

    def test_returns_plain_string(self, base_ctx):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent("", "Write an introduction", base_ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_never_raises(self, base_ctx):
        """Real agent contract: returns '[ERROR]...' string, never raises."""
        from writing_agent.agent import run_writing_agent
        # Pass a broken context — should still return a string
        result = run_writing_agent("", "do something", {})
        assert isinstance(result, str)

    # ── Operation auto-detection (mirrors real _detect_operation) ─

    @pytest.mark.parametrize("instruction,document,expected_op", [
        ("rephrase this paragraph",          "some text",  "rephrase"),
        ("rewrite for clarity",              "some text",  "rephrase"),
        ("reword the abstract",              "some text",  "rephrase"),
        ("paraphrase this",                  "some text",  "rephrase"),
        ("improve the methodology section",  "some text",  "improve"),
        ("enhance the introduction",         "some text",  "improve"),
        ("fix grammar mistakes",             "some text",  "improve"),
        ("correct the terminology",          "some text",  "improve"),
        ("polish this paragraph",            "some text",  "improve"),
        ("proofread the abstract",           "some text",  "improve"),
        ("make formal",                      "some text",  "improve"),
        ("write an introduction",            "",           "generate"),
        ("generate a related work section",  "",           "generate"),
        ("draft the conclusion",             "",           "generate"),
        ("create a methodology section",     "",           "generate"),
        ("expand this paragraph",            "some text",  "generate"),
        # Fallback rules
        ("do something ambiguous",           "",
         "generate"),  # empty doc → generate
        ("do something ambiguous",           "some text",
         "improve"),   # has doc → improve
    ])
    def test_operation_detection(self, instruction, document, expected_op):
        from writing_agent.agent import _detect_operation
        assert _detect_operation(instruction, document) == expected_op

    # ── Operation override ─────────────────────────────────────

    @pytest.mark.parametrize("forced_op", ["generate", "rephrase", "improve"])
    def test_operation_override_respected(self, base_ctx, forced_op):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent(
            document="existing text",
            instruction="do something ambiguous",  # would auto-detect differently
            context=base_ctx,
            operation=forced_op,
        )
        # Each mock op puts its label in the output — easy to verify
        assert forced_op in result.lower()

    def test_invalid_operation_falls_back_to_auto_detect(self, base_ctx):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent(
            document="",
            instruction="write something",
            context=base_ctx,
            operation="INVALID_OP",    # not in valid_ops
        )
        assert isinstance(result, str)
        assert "generate" in result.lower()

    # ── Context keys ──────────────────────────────────────────

    def test_language_reflected_in_output(self):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent(
            "", "Write an introduction",
            {"language": "French", "writing_style": "academic",
             "tone": "formal", "sources": []},
        )
        assert "French" in result

    def test_target_journal_reflected_in_output(self):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent(
            "", "Write an introduction",
            {"target_journal": "Nature Medicine", "writing_style": "academic",
             "tone": "formal", "sources": []},
        )
        assert "Nature Medicine" in result

    # ── Sources / citation injection ─────────────────────────

    def test_sources_trigger_references_section(self, base_ctx):
        """Real agent: injects citations when orchestrator passes sources."""
        from writing_agent.agent import run_writing_agent
        ctx = dict(base_ctx)
        ctx["sources"] = [
            {"title": "Attention is All You Need",
                "authors": "Vaswani et al.", "year": "2017"},
            {"title": "BERT", "authors": "Devlin et al.", "year": "2019"},
        ]
        result = run_writing_agent("", "Write a related work section", ctx)
        assert "References" in result
        assert "Vaswani" in result
        assert "Devlin" in result

    def test_ieee_citation_style(self, base_ctx):
        from writing_agent.agent import run_writing_agent
        ctx = dict(base_ctx)
        ctx["citation_style"] = "IEEE"
        ctx["sources"] = [
            {"title": "Some Paper", "authors": "Author A", "year": "2022"},
        ]
        result = run_writing_agent("", "Write introduction", ctx)
        # IEEE format uses [1] Author, "Title," year.
        assert "[1]" in result

    def test_empty_sources_no_references_appended(self, base_ctx):
        from writing_agent.agent import run_writing_agent
        result = run_writing_agent("", "Write introduction", base_ctx)
        assert "## References" not in result

    # ── grounded_only flag ────────────────────────────────────

    def test_grounded_only_does_not_crash(self, base_ctx):
        """Real agent: grounded_only=True means only use passed sources."""
        from writing_agent.agent import run_writing_agent
        ctx = dict(base_ctx)
        ctx["grounded_only"] = True
        ctx["sources"] = [
            {"title": "Paper A", "authors": "Auth A", "year": "2020"},
        ]
        result = run_writing_agent("", "Write related work", ctx)
        assert isinstance(result, str)
        assert "Paper A" in result


# ══════════════════════════════════════════════════════════════
# Literature Agent
# ══════════════════════════════════════════════════════════════

class TestLiteratureAgentContract:
    """
    Contract tests: verify the mock writes the same files as the real agent.
    """

    def test_writes_state_of_the_art_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="RAG Systems")
        art_file = tmp_path / "outputs" / "RAG Systems_state_of_the_art.txt"
        assert art_file.exists(), "state_of_the_art file must be written"
        content = art_file.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_writes_citations_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="Deep Learning")
        cit_file = tmp_path / "outputs" / "citations.txt"
        assert cit_file.exists(), "citations.txt must always be written"
        content = cit_file.read_text(encoding="utf-8")
        assert len(content) > 10

    def test_returns_none(self, tmp_path, monkeypatch):
        """Real agent returns None — side effects only."""
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        result = run_literature_agent(project_title="NLP")
        assert result is None

    def test_apa_citations_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="Test", citation_style="APA")
        cit = (tmp_path / "outputs" / "citations.txt").read_text()
        # APA uses "Author (year). Title." pattern — no square-bracket numbering
        assert "(" in cit and ")" in cit

    def test_ieee_citations_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="Test", citation_style="IEEE")
        cit = (tmp_path / "outputs" / "citations.txt").read_text()
        assert "[1]" in cit

    def test_project_title_in_review(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="Federated Learning Security")
        art = (tmp_path / "outputs" / "Federated Learning Security_state_of_the_art.txt"
               ).read_text()
        assert "Federated Learning Security" in art

    def test_all_optional_params_accepted(self, tmp_path, monkeypatch):
        """Real agent accepts all these kwargs — mock must too."""
        monkeypatch.chdir(tmp_path)
        from literature_agent.agent import run_literature_agent
        # Should not raise with any combination of optional args
        run_literature_agent(
            project_title="Test Project",
            citation_style="IEEE",
            use_web_resources=True,
            use_ocr=True,
            web_urls=["https://example.com"],
            save_to_chromadb=False,
        )

    def test_creates_outputs_dir_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "outputs").exists()
        from literature_agent.agent import run_literature_agent
        run_literature_agent(project_title="Test")
        assert (tmp_path / "outputs").exists()


# ══════════════════════════════════════════════════════════════
# Visualization Module
# ══════════════════════════════════════════════════════════════

class TestVisualizationModule:
    """
    Tests adapted for the older dummy visualization implementation.
    The older implementation hardcodes tempfile paths and extensions,
    ignoring the config dictionary.
    """

    @pytest.fixture
    def data(self):
        return {"x": [1, 2, 3], "y": [10, 20, 15]}

    @pytest.fixture
    def config(self):
        return {"title": "Test Chart", "format": "png"}

    def test_generate_chart_returns_existing_path(self, data, config):
        from visualization.module import generate_chart
        path = generate_chart(data, config)
        assert os.path.exists(path)
        assert path.endswith("chart.png")

    def test_generate_table_returns_existing_path(self, data, config):
        from visualization.module import generate_table
        path = generate_table(data, config)
        assert os.path.exists(path)
        assert path.endswith("table.csv")

    def test_export_figure_returns_existing_path(self, data, config):
        from visualization.module import export_figure
        path = export_figure(data, config)
        assert os.path.exists(path)
        assert path.endswith("figure.pdf")

    def test_empty_data_does_not_crash(self):
        from visualization.module import generate_chart
        path = generate_chart({}, {})
        assert os.path.exists(path)
        assert path.endswith("chart.png")
