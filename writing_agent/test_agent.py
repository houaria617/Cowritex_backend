"""
test_agent_cli.py — Interactive console tester for the Writing Agent

Usage:
    python test_agent_cli.py               # interactive mode
    python test_agent_cli.py --quick       # section-writing tests
    python test_agent_cli.py --suggestions # copilot suggestion tests
    python test_agent_cli.py --task-detect # test auto task extraction
"""

import sys
import os
import argparse
import textwrap
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESET  = "\033[0m";  BOLD   = "\033[1m"
GREEN  = "\033[92m"; CYAN   = "\033[96m"
YELLOW = "\033[93m"; RED    = "\033[91m"
DIM    = "\033[2m";  BLUE   = "\033[94m"

def c(text, color): return f"{color}{text}{RESET}"

def banner():
    print(c("""
╔══════════════════════════════════════════════════════════╗
║         CoWriteX — Writing Agent  CLI Tester             ║
║         Interactive Console  •  agent.py  v3.0           ║
║  NEW: section targeting · auto task · surrounding aware  ║
╚══════════════════════════════════════════════════════════╝
""", CYAN))

def section(title):
    print(f"\n{c('─' * 60, DIM)}")
    print(c(f"  {title}", BOLD + BLUE))
    print(c('─' * 60, DIM))

def print_result(result: str):
    print(c("\n📄 OUTPUT:", GREEN + BOLD))
    print(c("┌" + "─" * 58 + "┐", DIM))
    for line in textwrap.wrap(result, width=56):
        print(c("│ ", DIM) + line)
    print(c("└" + "─" * 58 + "┘", DIM))
    print(c(f"  ↳ {len(result.split())} words generated", DIM))

def print_suggestion(result: dict):
    if "error" in result:
        print(c(f"\n❌ Suggestion error: {result['error']}", RED))
        return
    print(c(f"\n💡 SUGGESTION  [{result.get('mode', '?')}]:", CYAN + BOLD))
    print(c(f"  ORIGINAL  : {result['original'] or '(completion — no original)'}", DIM))
    print(c(f"  SUGGESTED : {result['suggestion']}", GREEN))
    diff = result.get("diff", [])
    if diff:
        print(c("\n  DIFF:", YELLOW))
        diff_str = ""
        for chunk in diff:
            if chunk["type"] == "equal":
                diff_str += chunk["text"] + " "
            elif chunk["type"] == "remove":
                diff_str += c(f"[-{chunk['text']}-]", RED) + " "
            elif chunk["type"] == "add":
                diff_str += c(f"[+{chunk['text']}+]", GREEN) + " "
        print("  " + diff_str.strip())


# ─────────────────────────────────────────────
# SHARED SAMPLE CONTEXT
# ─────────────────────────────────────────────

_CTX_IEEE = {
    "writing_style": "academic", "tone": "formal",
    "language": "English", "citation_style": "IEEE",
    "target_journal": "IEEE Transactions on Neural Networks",
    "grounded_only": False, "sources": [],
}

_CTX_APA = {
    "writing_style": "academic", "tone": "formal",
    "language": "English", "citation_style": "APA",
    "grounded_only": False, "sources": [],
}

# Shared sample surrounding sections used across several tests
_ABSTRACT_TEXT = (
    "This paper introduces CoWriteX, a human-in-the-loop AI writing assistant "
    "designed to help researchers draft, refine, and structure research paper sections "
    "using large language models. Experimental evaluation shows significant improvements "
    "in both writing quality (BLEU +12%) and researcher productivity (time-to-draft –38%)."
)

_RELATED_WORK_TEXT = (
    "Prior systems such as GPT-3-based editors and Grammarly focus on surface-level "
    "grammar corrections. Retrieval-augmented generation (RAG) has been applied to "
    "question answering but not to full academic section writing. Our system bridges "
    "this gap by combining structured section blueprints with real-time LLM suggestions."
)

_METHODOLOGY_TEXT = (
    "The proposed system fine-tunes LLaMA-3-70B on 120k academic paper sections "
    "extracted from arXiv. We use a retrieval-augmented pipeline where ChromaDB stores "
    "literature chunks, and a LangChain orchestrator routes each writing request to the "
    "appropriate prompt template based on the detected section type."
)


# ─────────────────────────────────────────────
# QUICK TESTS — section writing
# ─────────────────────────────────────────────

QUICK_TESTS = [
    # ── Test 1: Explicit section + preceding/next sections ────────────────
    {
        "name":            "Generate — Introduction (section-aware, no-repeat)",
        "document":        "",
        "instruction":     "Write an introduction for a paper about RAG-based AI writing assistants.",
        "context":         _CTX_IEEE,
        "operation":       None,           # auto-detected
        "target_section":  "introduction",
        "preceding_sections": {"abstract": _ABSTRACT_TEXT},
        "next_sections":      {"related work": _RELATED_WORK_TEXT},
    },
    # ── Test 2: Abstract with surrounding awareness ───────────────────────
    {
        "name":            "Generate — Abstract (explicit section, next-section aware)",
        "document":        "",
        "instruction":     "Write an abstract for a paper proposing CoWriteX, an AI writing assistant for researchers.",
        "context":         _CTX_IEEE,
        "operation":       "generate",
        "target_section":  "abstract",
        "preceding_sections": None,
        "next_sections":      {"introduction": _RELATED_WORK_TEXT},
    },
    # ── Test 3: Auto task extraction — rephrase detected from free text ───
    {
        "name":            "Rephrase — auto task extraction from instruction",
        "document":        "AI is useful. It helps people write papers. The system is good at fixing mistakes.",
        "instruction":     "Can you reword this to sound more formal and academic?",
        "context":         _CTX_APA,
        "operation":       None,           # ← no explicit op; agent extracts from instruction
        "target_section":  "introduction",
        "preceding_sections": None,
        "next_sections":      None,
    },
    # ── Test 4: Auto task extraction — improve detected ───────────────────
    {
        "name":            "Improve — auto task extraction from instruction",
        "document":        "The methodology uses a transformer model fine-tuned on academic data.",
        "instruction":     "Polish this paragraph and fix the grammar. Make it more precise.",
        "context":         _CTX_IEEE,
        "operation":       None,           # ← extracted as "improve"
        "target_section":  "methodology",
        "preceding_sections": {"related work": _RELATED_WORK_TEXT},
        "next_sections":      {"results": "We evaluated on three benchmarks: BLEU, ROUGE, and BERTScore."},
    },
    # ── Test 5: Injected sources + surrounding awareness ─────────────────
    {
        "name":            "Generate — Related Work (injected sources + surrounding)",
        "document":        "",
        "instruction":     "Write a related work section about transformer models in NLP.",
        "context": {
            **_CTX_IEEE, "grounded_only": True,
            "sources": [
                {
                    "title": "Attention is All You Need", "authors": "Vaswani et al.", "year": "2017",
                    "abstract": "We propose the Transformer, based entirely on attention mechanisms.",
                    "source_file": "vaswani2017.pdf", "page": 1,
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": "Devlin et al.", "year": "2019",
                    "abstract": "BERT pre-trains deep bidirectional representations from unlabeled text.",
                    "source_file": "devlin2019.pdf", "page": 1,
                },
            ],
        },
        "operation":       "generate",
        "target_section":  "related work",
        "preceding_sections": {"introduction": _ABSTRACT_TEXT},
        "next_sections":      {"methodology": _METHODOLOGY_TEXT},
    },
]


# ─────────────────────────────────────────────
# SUGGESTION TESTS
# ─────────────────────────────────────────────

SUGGESTION_TESTS = [
    # ── Test 1: Mode auto-detected (improve) + section aware ─────────────
    {
        "name":            "Suggestion — auto-detect mode + section-aware (no surrounding repeat)",
        "document":        "This paper presents CoWriteX, an AI writing assistant using LLMs.",
        "target_text":     "The method works well on all datasets.",
        "context":         _CTX_IEEE,
        "mode":            None,               # ← auto-detected
        "target_section":  "results",
        "preceding_sections": {"methodology": _METHODOLOGY_TEXT},
        "next_sections":      {"discussion": "We discuss the implications of these results below."},
    },
    # ── Test 2: Complete (no selection, cursor at end) ────────────────────
    {
        "name":            "Suggestion — complete from cursor (no selection)",
        "document":        "In recent years, large language models have demonstrated remarkable capabilities in natural language understanding. However, their application to",
        "target_text":     "",
        "context":         _CTX_APA,
        "mode":            None,               # → auto-detected as "complete"
        "target_section":  "introduction",
        "preceding_sections": {"abstract": _ABSTRACT_TEXT},
        "next_sections":      None,
    },
    # ── Test 3: Rephrase paragraph (long selection → auto rephrase) ───────
    {
        "name":            "Suggestion — auto-detect rephrase (long selection)",
        "document":        "Transformer models have revolutionised NLP.",
        "target_text":     (
            "Transformer models are really good at NLP tasks. "
            "They use attention to look at all parts of the input at once. "
            "This makes them better than RNNs which process tokens one by one. "
            "Also they scale better with more data and compute resources available."
        ),
        "context":         _CTX_APA,
        "mode":            None,               # → auto-detected as "rephrase" (>40 words)
        "target_section":  "related work",
        "preceding_sections": {"introduction": _ABSTRACT_TEXT},
        "next_sections":      {"methodology": _METHODOLOGY_TEXT},
    },
    # ── Test 4: Explicit mode override (improve) ──────────────────────────
    {
        "name":            "Suggestion — explicit improve mode",
        "document":        "The proposed architecture consists of three main components.",
        "target_text":     "The first part does the encoding.",
        "context":         _CTX_IEEE,
        "mode":            "improve",          # ← explicit
        "target_section":  "methodology",
        "preceding_sections": None,
        "next_sections":      None,
    },
]


# ─────────────────────────────────────────────
# TASK-DETECTION TESTS  (no API call needed)
# ─────────────────────────────────────────────

TASK_DETECTION_CASES = [
    ("Can you make this paragraph more formal and fix the grammar?",         "improve"),
    ("Please reword the following sentence to be clearer.",                   "rephrase"),
    ("Draft a methodology section for my paper on federated learning.",       "generate"),
    ("Polish this abstract and tighten the language.",                        "improve"),
    ("Rephrase this results paragraph to avoid passive voice.",               "rephrase"),
    ("Write an introduction about transformer models in healthcare AI.",      "generate"),
    ("Could you proofread this and fix any academic style issues?",           "improve"),
    ("Paraphrase this to be more concise.",                                   "rephrase"),
    ("Compose a discussion section based on these results.",                  "generate"),
    ("Strengthen the contribution statement in the introduction.",            "improve"),
]


def run_task_detection_tests():
    try:
        from agent import _detect_operation
    except ImportError as e:
        print(c(f"\n[ERROR] {e}", RED)); return

    banner()
    section("Auto Task Extraction — Pattern Tests (no API)")
    passed = failed = 0
    for instr, expected in TASK_DETECTION_CASES:
        detected = _detect_operation(instr, "some existing text")
        ok = detected == expected
        mark = c("✓", GREEN) if ok else c("✗", RED)
        label = c(f"[{detected}]", GREEN if ok else RED)
        exp   = c(f"expected [{expected}]", DIM)
        print(f"  {mark} {label} {exp}")
        print(c(f"     → \"{instr[:65]}\"", DIM))
        if ok: passed += 1
        else:  failed += 1
    print(c(f"\n  Results: {passed} passed  |  {failed} failed\n", BOLD))


# ─────────────────────────────────────────────
# TEST RUNNERS
# ─────────────────────────────────────────────

def run_writing_tests(tests: list):
    try:
        from agent import run_writing_agent
    except ImportError as e:
        print(c(f"\n[ERROR] {e}", RED)); return

    print(c(f"\n🚀 Running {len(tests)} writing test(s)\n", BOLD + GREEN))
    passed = failed = 0
    for i, t in enumerate(tests, 1):
        section(f"Test {i}/{len(tests)} — {t['name']}")
        if t["document"]:
            print(c(f'  TEXT: "{t["document"][:90]}..."', DIM))
        print(c(f"  INSTRUCTION   : {t['instruction']}", BOLD))
        print(c(f"  TARGET SECTION: {t.get('target_section') or 'auto-detect'}", CYAN))
        print(c(f"  PRECEDING     : {list(t.get('preceding_sections') or {})}", DIM))
        print(c(f"  NEXT          : {list(t.get('next_sections') or {})}", DIM))
        try:
            result = run_writing_agent(
                document            = t["document"],
                instruction         = t["instruction"],
                context             = t["context"],
                operation           = t.get("operation"),
                target_section      = t.get("target_section"),
                preceding_sections  = t.get("preceding_sections"),
                next_sections       = t.get("next_sections"),
            )
            if result.startswith("[ERROR]"):
                print(c(f"\n❌ {result}", RED)); failed += 1
            else:
                print_result(result); passed += 1
        except Exception as exc:
            print(c(f"\n❌ EXCEPTION: {exc}", RED))
            import traceback; traceback.print_exc()
            failed += 1

    print(c(f"\n{'═'*60}\n  Results: {passed} passed  |  {failed} failed\n{'═'*60}\n", DIM))


def run_suggestion_tests(tests: list):
    try:
        from agent import run_suggestion_agent
    except ImportError as e:
        print(c(f"\n[ERROR] {e}", RED)); return

    print(c(f"\n💡 Running {len(tests)} suggestion test(s)\n", BOLD + CYAN))
    passed = failed = 0
    for i, t in enumerate(tests, 1):
        section(f"Suggestion {i}/{len(tests)} — {t['name']}")
        print(c(f"  DOCUMENT      : {t['document'][:70]}...", DIM))
        print(c(f"  TARGET SECTION: {t.get('target_section') or 'auto-detect'}", CYAN))
        if t["target_text"]:
            print(c(f"  TARGET TEXT   : {t['target_text'][:70]}", BOLD))
        else:
            print(c("  TARGET TEXT   : (no selection — completion mode)", DIM))
        print(c(f"  MODE          : {t['mode'] or 'auto-detect'}", DIM))
        print(c(f"  PRECEDING     : {list(t.get('preceding_sections') or {})}", DIM))
        print(c(f"  NEXT          : {list(t.get('next_sections') or {})}", DIM))

        try:
            result = run_suggestion_agent(
                document            = t["document"],
                context             = t["context"],
                target_text         = t["target_text"],
                suggestion_mode     = t["mode"],
                target_section      = t.get("target_section"),
                preceding_sections  = t.get("preceding_sections"),
                next_sections       = t.get("next_sections"),
            )
            if "error" in result:
                print(c(f"\n❌ {result['error']}", RED)); failed += 1
            else:
                print_suggestion(result); passed += 1
        except Exception as exc:
            print(c(f"\n❌ EXCEPTION: {exc}", RED))
            import traceback; traceback.print_exc()
            failed += 1

    print(c(f"\n{'═'*60}\n  Results: {passed} passed  |  {failed} failed\n{'═'*60}\n", DIM))


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

_KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "literature review",
    "methodology", "results", "discussion", "conclusion",
]

def interactive_mode():
    try:
        from agent import run_writing_agent, run_suggestion_agent
    except ImportError as e:
        print(c(f"\n[ERROR] {e}", RED)); sys.exit(1)

    banner()
    session = 0
    while True:
        try:
            session += 1
            section(f"Session #{session}")
            print(c("  Mode: (1) Write/Improve section  (2) Get inline suggestion", YELLOW))
            mode_pick = input(c("  Choose [1]: ", YELLOW)).strip()

            if mode_pick == "2":
                # ── Suggestion mode ─────────────────────────────────
                document    = input(c("\n  Full document (or paste snippet): ", CYAN)).strip()
                target_text = input(c("  Selected text (or Enter to complete): ", CYAN)).strip()

                # Section picker
                print(c("\n  Target section:", BOLD))
                for i, s in enumerate(_KNOWN_SECTIONS, 1):
                    print(f"  {c(str(i), CYAN)}. {s}")
                sec_raw = input(c("  Choose section number (or type name, Enter=auto): ", YELLOW)).strip()
                if sec_raw.isdigit() and 1 <= int(sec_raw) <= len(_KNOWN_SECTIONS):
                    target_sec = _KNOWN_SECTIONS[int(sec_raw) - 1]
                else:
                    target_sec = sec_raw or None

                prec_raw = input(c("\n  Preceding section text (Enter to skip): ", YELLOW)).strip()
                preceding = {_KNOWN_SECTIONS[0]: prec_raw} if prec_raw else None
                next_raw  = input(c("  Next section text (Enter to skip): ", YELLOW)).strip()
                nxt       = {_KNOWN_SECTIONS[-1]: next_raw} if next_raw else None

                ctx = {**{"writing_style": "academic", "tone": "formal",
                           "language": "English", "citation_style": "IEEE",
                           "grounded_only": False, "sources": []}}
                result = run_suggestion_agent(
                    document=document, context=ctx,
                    target_text=target_text,
                    target_section=target_sec,
                    preceding_sections=preceding,
                    next_sections=nxt,
                )
                print_suggestion(result)

            else:
                # ── Writing mode ────────────────────────────────────
                instruction = input(c("\n✏  Instruction (free text — task is auto-extracted): ", CYAN + BOLD)).strip()

                # Section picker
                print(c("\n  Target section:", BOLD))
                for i, s in enumerate(_KNOWN_SECTIONS, 1):
                    print(f"  {c(str(i), CYAN)}. {s}")
                sec_raw = input(c("  Choose section number (or type name, Enter=auto): ", YELLOW)).strip()
                if sec_raw.isdigit() and 1 <= int(sec_raw) <= len(_KNOWN_SECTIONS):
                    target_sec = _KNOWN_SECTIONS[int(sec_raw) - 1]
                else:
                    target_sec = sec_raw or None

                lines = []
                print(c("\n📝 Existing text (Enter twice to skip):", YELLOW))
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                document = "\n".join(lines).strip()

                prec_raw = input(c("\n  Preceding section text (Enter to skip): ", YELLOW)).strip()
                preceding = {"abstract": prec_raw} if prec_raw else None
                next_raw  = input(c("  Next section text (Enter to skip): ", YELLOW)).strip()
                nxt       = {"related work": next_raw} if next_raw else None

                ctx = {"writing_style": "academic", "tone": "formal",
                       "language": "English", "citation_style": "IEEE",
                       "grounded_only": False, "sources": []}
                result = run_writing_agent(
                    document=document, instruction=instruction,
                    context=ctx, operation=None,
                    target_section=target_sec,
                    preceding_sections=preceding,
                    next_sections=nxt,
                )
                if result.startswith("[ERROR]"):
                    print(c(f"\n❌ {result}", RED))
                else:
                    print_result(result)

            if input(c("\n  Run another? (Y/n): ", YELLOW)).strip().lower() == "n":
                print(c("\n  Bye! 👋\n", GREEN)); break
        except KeyboardInterrupt:
            print(c("\n\n  Interrupted. Bye! 👋\n", GREEN)); break


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoWriteX Writing Agent — CLI Tester v3.0")
    parser.add_argument("--quick",       action="store_true", help="Section-writing tests with surrounding awareness")
    parser.add_argument("--suggestions", action="store_true", help="Copilot suggestion tests")
    parser.add_argument("--task-detect", action="store_true", help="Test auto task extraction (no API needed)")
    args = parser.parse_args()

    banner()
    if args.quick:
        run_writing_tests(QUICK_TESTS)
    elif args.suggestions:
        run_suggestion_tests(SUGGESTION_TESTS)
    elif args.task_detect:
        run_task_detection_tests()
    else:
        interactive_mode()

if __name__ == "__main__":
    main()