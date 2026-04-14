"""
test_agent_cli.py — Interactive console tester for the Writing Agent

Usage:
    python test_agent_cli.py               # interactive mode
    python test_agent_cli.py --quick       # run built-in quick tests
    python test_agent_cli.py --demo        # run full demo suite (no API needed)
"""

import sys
import os
import argparse
import textwrap

# ── Allow running from any directory ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BLUE   = "\033[94m"

def c(text, color):
    return f"{color}{text}{RESET}"

def banner():
    print(c("""
╔══════════════════════════════════════════════════════════╗
║         CoWriteX — Writing Agent  CLI Tester             ║
║         Interactive Console  •  agent.py  v1.0           ║
╚══════════════════════════════════════════════════════════╝
""", CYAN))

def section(title):
    print(f"\n{c('─' * 60, DIM)}")
    print(c(f"  {title}", BOLD + BLUE))
    print(c('─' * 60, DIM))

def print_result(result: str):
    print(c("\n📄 OUTPUT:", GREEN + BOLD))
    print(c("┌" + "─" * 58 + "┐", DIM))
    wrapped = textwrap.fill(result, width=56)
    for line in wrapped.split("\n"):
        print(c("│ ", DIM) + line)
    print(c("└" + "─" * 58 + "┘", DIM))
    words = len(result.split())
    print(c(f"  ↳ {words} words generated", DIM))

def pick_style() -> dict:
    """Walk the user through writing-preference prompts."""
    print(c("\n⚙  Configure writing preferences (Enter = keep default):", YELLOW))

    styles = ["academic", "technical", "expository", "narrative"]
    tones  = ["formal", "semi-formal", "neutral", "conversational"]
    cites  = ["APA", "IEEE", "MLA", "Chicago", "None"]
    langs  = ["English", "French", "Arabic", "Spanish", "German"]

    def _pick(prompt, options, default):
        for i, o in enumerate(options, 1):
            marker = c(" ← default", DIM) if o == default else ""
            print(f"  {c(str(i), CYAN)}. {o}{marker}")
        raw = input(c(f"  {prompt} [{default}]: ", YELLOW)).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        return default

    print(c("\n  Writing style:", BOLD))
    style = _pick("Choose style", styles, "academic")

    print(c("\n  Tone:", BOLD))
    tone = _pick("Choose tone", tones, "formal")

    print(c("\n  Citation style:", BOLD))
    cite = _pick("Choose citations", cites, "IEEE")

    print(c("\n  Language:", BOLD))
    lang = _pick("Choose language", langs, "English")

    journal = input(c("\n  Target journal (optional, Enter to skip): ", YELLOW)).strip()
    grounded = input(c("  Strict grounding — only use provided sources? (y/N): ", YELLOW)).strip().lower()

    return {
        "writing_style":  style,
        "tone":           tone,
        "citation_style": cite,
        "language":       lang,
        "target_journal": journal,
        "grounded_only":  grounded == "y",
        "sources":        [],
    }


def pick_operation() -> str | None:
    print(c("\n  Operation override (optional):", BOLD))
    ops = ["auto-detect", "generate", "rephrase", "improve"]
    for i, o in enumerate(ops, 1):
        print(f"  {c(str(i), CYAN)}. {o}")
    raw = input(c("  Choose [1=auto]: ", YELLOW)).strip()
    if raw == "2": return "generate"
    if raw == "3": return "rephrase"
    if raw == "4": return "improve"
    return None   # auto-detect


# ─────────────────────────────────────────────
# BUILT-IN TEST SUITES
# ─────────────────────────────────────────────

QUICK_TESTS = [
    {
        "name":        "Generate — Introduction",
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
        "name":        "Rephrase — Make formal",
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
        "name":        "Generate — French, APA",
        "document":    "",
        "instruction": "Write an introduction about deep learning in medical imaging.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "French", "citation_style": "APA",
            "grounded_only": False, "sources": [],
        },
        "operation": None,
    },
    {
        "name":        "Related Work — with injected sources",
        "document":    "",
        "instruction": "Write a related work section about transformer models in NLP.",
        "context": {
            "writing_style": "academic", "tone": "formal",
            "language": "English", "citation_style": "IEEE",
            "grounded_only": True,
            "sources": [
                {
                    "title":       "Attention is All You Need",
                    "authors":     "Vaswani et al.",
                    "year":        "2017",
                    "abstract":    "We propose the Transformer, a model architecture based entirely "
                                   "on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                    "source_file": "vaswani2017.pdf",
                    "page":        1,
                },
                {
                    "title":       "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors":     "Devlin et al.",
                    "year":        "2019",
                    "abstract":    "We introduce BERT, designed to pre-train deep bidirectional "
                                   "representations from unlabeled text by jointly conditioning on context.",
                    "source_file": "devlin2019.pdf",
                    "page":        1,
                },
            ],
        },
        "operation": "generate",
    },
]


def run_tests(tests: list, label: str = "Test Suite"):
    try:
        from agent import run_writing_agent
    except ImportError as e:
        print(c(f"\n[ERROR] Could not import agent.py: {e}", RED))
        print(c("Make sure agent.py, prompts.py, and tools.py are in the same directory.", YELLOW))
        return

    print(c(f"\n🚀 Running {label} — {len(tests)} test(s)\n", BOLD + GREEN))

    passed = failed = 0
    for i, t in enumerate(tests, 1):
        section(f"Test {i}/{len(tests)} — {t['name']}")

        if t["document"]:
            print(c("  CURRENT TEXT:", DIM))
            print(c(f'  "{t["document"][:120]}..."' if len(t["document"]) > 120 else f'  "{t["document"]}"', DIM))

        print(c(f"  INSTRUCTION: {t['instruction']}", BOLD))
        op_label = t.get("operation") or "auto"
        print(c(f"  OPERATION:   {op_label}", DIM))

        try:
            result = run_writing_agent(
                document    = t["document"],
                instruction = t["instruction"],
                context     = t["context"],
                operation   = t.get("operation"),
            )

            if result.startswith("[ERROR]"):
                print(c(f"\n❌ AGENT ERROR: {result}", RED))
                failed += 1
            else:
                print_result(result)
                passed += 1

        except Exception as exc:
            print(c(f"\n❌ EXCEPTION: {exc}", RED))
            import traceback; traceback.print_exc()
            failed += 1

    print(c(f"\n{'═' * 60}", DIM))
    print(c(f"  Results: {passed} passed  |  {failed} failed", BOLD))
    print(c(f"{'═' * 60}\n", DIM))


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def interactive_mode():
    try:
        from agent import run_writing_agent
    except ImportError as e:
        print(c(f"\n[ERROR] Could not import agent.py: {e}", RED))
        print(c("Make sure agent.py, prompts.py, and tools.py are in the same directory.\n", YELLOW))
        sys.exit(1)

    banner()
    print(c("  Type your instruction and current text interactively.", DIM))
    print(c("  Press Ctrl+C at any time to quit.\n", DIM))

    session = 0
    while True:
        try:
            session += 1
            section(f"Session #{session}")

            # Instruction
            instruction = ""
            while not instruction.strip():
                instruction = input(c("\n✏  Instruction: ", CYAN + BOLD)).strip()
                if not instruction:
                    print(c("  Instruction cannot be empty.", RED))

            # Existing document (optional)
            print(c("\n📝 Existing text (optional — press Enter twice to skip):", YELLOW))
            lines = []
            while True:
                try:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            document = "\n".join(lines).strip()

            # Preferences
            use_defaults = input(c("\n  Use default preferences? (Y/n): ", YELLOW)).strip().lower()
            if use_defaults == "n":
                ctx = pick_style()
                operation = pick_operation()
            else:
                ctx = {
                    "writing_style": "academic", "tone": "formal",
                    "language": "English", "citation_style": "IEEE",
                    "grounded_only": False, "sources": [],
                }
                operation = None

            print(c("\n⏳ Running Writing Agent...", YELLOW))

            result = run_writing_agent(
                document    = document,
                instruction = instruction,
                context     = ctx,
                operation   = operation,
            )

            if result.startswith("[ERROR]"):
                print(c(f"\n❌ Agent returned an error:\n{result}", RED))
            else:
                print_result(result)

            again = input(c("\n  Run another? (Y/n): ", YELLOW)).strip().lower()
            if again == "n":
                print(c("\n  Bye! 👋\n", GREEN))
                break

        except KeyboardInterrupt:
            print(c("\n\n  Interrupted. Bye! 👋\n", GREEN))
            break


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoWriteX Writing Agent — CLI Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python test_agent_cli.py               # full interactive mode
              python test_agent_cli.py --quick       # 5 built-in quick tests
        """),
    )
    parser.add_argument("--quick", action="store_true", help="Run 5 built-in quick tests")
    args = parser.parse_args()

    if args.quick:
        banner()
        run_tests(QUICK_TESTS, label="Quick Test Suite")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()