"""
test_agent_cli.py — Interactive console tester for the Writing Agent

Usage:
    python test_agent_cli.py               # interactive mode
    python test_agent_cli.py --quick       # 5 section-writing tests
    python test_agent_cli.py --suggestions # 4 copilot suggestion tests
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
║         Interactive Console  •  agent.py  v2.0           ║
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
    """Pretty-print a suggestion dict from run_suggestion_agent."""
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
# QUICK TESTS — section writing
# ─────────────────────────────────────────────

QUICK_TESTS = [
    {
        "name":        "Generate — structured Introduction",
        "document":    "",
        "instruction": "Write an introduction for a paper about RAG-based AI writing assistants for researchers.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "target_journal": "IEEE Transactions on Neural Networks",
            "grounded_only": False, "sources": [],
        },
        "operation": None,
    },
    {
        "name":        "Generate — structured Abstract",
        "document":    "",
        "instruction": "Write an abstract for a paper proposing a transformer-based model for automatic code documentation.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "target_journal": "ACM Transactions on Software Engineering",
            "grounded_only": False, "sources": [],
        },
        "operation": "generate",
    },
    {
        "name":        "Rephrase — make formal",
        "document":    "AI is useful. It helps people write papers. The system is good at fixing mistakes.",
        "instruction": "Rephrase this text to be more formal and academic.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "APA",
            "grounded_only": False, "sources": [],
        },
        "operation": "rephrase",
    },
    {
        "name":        "Improve — Methodology paragraph",
        "document":    "The methodology uses a transformer model fine-tuned on academic data.",
        "instruction": "Improve this methodology paragraph with more technical detail.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "target_journal": "Nature Machine Intelligence",
            "grounded_only": False, "sources": [],
        },
        "operation": "improve",
    },
    {
        "name":        "Generate — Related Work with injected sources",
        "document":    "",
        "instruction": "Write a related work section about transformer models in NLP.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "grounded_only": True,
            "sources": [
                {
                    "title": "Attention is All You Need", "authors": "Vaswani et al.", "year": "2017",
                    "abstract": "We propose the Transformer, a model architecture based entirely "
                                "on attention mechanisms, dispensing with recurrence and convolutions.",
                    "source_file": "vaswani2017.pdf", "page": 1,
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": "Devlin et al.", "year": "2019",
                    "abstract": "We introduce BERT, designed to pre-train deep bidirectional "
                                "representations from unlabeled text by jointly conditioning on context.",
                    "source_file": "devlin2019.pdf", "page": 1,
                },
            ],
        },
        "operation": "generate",
    },
]


# ─────────────────────────────────────────────
# SUGGESTION TESTS
# ─────────────────────────────────────────────

SUGGESTION_TESTS = [
    {
        "name":        "Suggestion — improve weak sentence",
        "document":    "This paper presents a new method for academic writing assistance using LLMs.",
        "target_text": "The method works well on all datasets.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "target_journal": "IEEE Transactions on Neural Networks",
            "grounded_only": False, "sources": [],
        },
        "mode": "improve",
    },
    {
        "name":        "Suggestion — complete a half-sentence (no selection)",
        "document":    "In recent years, large language models have demonstrated remarkable capabilities in natural language understanding and generation. However, their application to",
        "target_text": "",    # cursor at end, no text selected
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "APA",
            "grounded_only": False, "sources": [],
        },
        "mode": "complete",
    },
    {
        "name":        "Suggestion — rephrase a paragraph",
        "document":    "Transformer models have revolutionised NLP.",
        "target_text": (
            "Transformer models are really good at NLP tasks. "
            "They use attention to look at all parts of the input at once. "
            "This makes them better than RNNs which process tokens one by one."
        ),
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "APA",
            "grounded_only": False, "sources": [],
        },
        "mode": "rephrase",
    },
    {
        "name":        "Suggestion — auto-detect mode (no mode given)",
        "document":    "The proposed architecture consists of three main components.",
        "target_text": "The first part does the encoding.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "grounded_only": False, "sources": [],
        },
        "mode": None,   # ← auto-detect
    },
]


# ─────────────────────────────────────────────
# TEST RUNNERS
# ─────────────────────────────────────────────

def run_writing_tests(tests: list):
    try:
        from agent import run_writing_agent
    except ImportError as e:
        print(c(f"\n[ERROR] Could not import agent.py: {e}", RED)); return

    print(c(f"\n🚀 Running {len(tests)} writing test(s)\n", BOLD + GREEN))
    passed = failed = 0
    for i, t in enumerate(tests, 1):
        section(f"Test {i}/{len(tests)} — {t['name']}")
        if t["document"]:
            print(c(f'  TEXT: "{t["document"][:100]}..."', DIM))
        print(c(f"  INSTRUCTION: {t['instruction']}", BOLD))
        try:
            result = run_writing_agent(
                document=t["document"], instruction=t["instruction"],
                context=t["context"], operation=t.get("operation"),
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
        print(c(f"\n[ERROR] Could not import agent.py: {e}", RED)); return

    print(c(f"\n💡 Running {len(tests)} suggestion test(s)\n", BOLD + CYAN))
    passed = failed = 0
    for i, t in enumerate(tests, 1):
        section(f"Suggestion {i}/{len(tests)} — {t['name']}")
        print(c(f"  DOCUMENT : {t['document'][:80]}...", DIM))
        if t["target_text"]:
            print(c(f"  TARGET   : {t['target_text'][:80]}", BOLD))
        else:
            print(c("  TARGET   : (no selection — completion mode)", DIM))
        print(c(f"  MODE     : {t['mode'] or 'auto-detect'}", DIM))

        try:
            result = run_suggestion_agent(
                document=t["document"],
                context=t["context"],
                target_text=t["target_text"],
                suggestion_mode=t["mode"],
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
                document = input(c("\n  Full document so far (or paste snippet): ", CYAN)).strip()
                target   = input(c("  Selected text (or Enter to complete from cursor): ", CYAN)).strip()
                modes    = ["auto", "complete", "improve", "rephrase"]
                for i, m in enumerate(modes):
                    print(f"  {c(str(i+1), CYAN)}. {m}")
                sm = input(c("  Suggestion mode [1=auto]: ", YELLOW)).strip()
                sm_map = {"2": "complete", "3": "improve", "4": "rephrase"}
                ctx = {
                    "writing_style": "academic", "tone": "formal",
                    "language": "English", "citation_style": "IEEE",
                    "grounded_only": False, "sources": [],
                }
                result = run_suggestion_agent(
                    document=document, context=ctx,
                    target_text=target, suggestion_mode=sm_map.get(sm),
                )
                print_suggestion(result)
            else:
                # ── Writing mode ────────────────────────────────────
                instruction = input(c("\n✏  Instruction: ", CYAN + BOLD)).strip()
                lines = []
                print(c("\n📝 Existing text (Enter twice to skip):", YELLOW))
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                document = "\n".join(lines).strip()
                ctx = {
                    "writing_style": "academic", "tone": "formal",
                    "language": "English", "citation_style": "IEEE",
                    "grounded_only": False, "sources": [],
                }
                result = run_writing_agent(
                    document=document, instruction=instruction,
                    context=ctx, operation=None,
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
    parser = argparse.ArgumentParser(description="CoWriteX Writing Agent — CLI Tester")
    parser.add_argument("--quick",       action="store_true", help="5 section-writing tests")
    parser.add_argument("--suggestions", action="store_true", help="4 copilot suggestion tests")
    args = parser.parse_args()

    banner()
    if args.quick:
        run_writing_tests(QUICK_TESTS)
    elif args.suggestions:
        run_suggestion_tests(SUGGESTION_TESTS)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()