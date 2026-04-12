"""
prompts.py — Writing Agent Prompt Templates (LangChain)

Uses LangChain's ChatPromptTemplate so prompts are clean,
inspectable, and easy to swap without touching agent logic.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic writing assistant embedded inside a \
human-in-the-loop research paper writing platform called CoWriteX.

Your role is to help researchers write, improve, and refine sections of their research papers.
You always follow the researcher's explicit instruction and adapt your output to their \
preferred writing style, tone, and target journal.

RULES YOU MUST FOLLOW:
1. Write ONLY what is asked — rephrase means rephrase; generate means write from scratch.
2. Preserve the researcher's voice and meaning when editing existing text.
3. If literature sources are provided, support claims with [SOURCE N] markers.
4. If grounded_only is True, do NOT add facts not supported by the provided sources.
5. Never add placeholders like "[insert reference here]" or "[TODO]".
6. Output clean publication-ready prose — no preamble, no meta-commentary.
7. Respect the target journal conventions when specified.
8. Always write in the specified language.
"""

# ─────────────────────────────────────────────
# SECTION-SPECIFIC GUIDANCE
# ─────────────────────────────────────────────

SECTION_GUIDANCE = {
    "abstract": (
        "Cover: (1) problem/motivation, (2) approach, (3) key results, (4) impact. "
        "Max 250 words. Past tense for what was done, present for conclusions."
    ),
    "introduction": (
        "Structure: (1) context & motivation, (2) literature gap, "
        "(3) paper contribution, (4) paper outline. Cite with [SOURCE N]."
    ),
    "methodology": (
        "Be precise and reproducible. Cover: dataset, preprocessing, model/algorithm, "
        "evaluation metrics, experimental setup. Use passive voice for formal journals."
    ),
    "results": (
        "Present results objectively with specific numbers. "
        "Compare against baselines. Do not interpret — save that for Discussion."
    ),
    "discussion": (
        "Interpret results relative to the research question and prior work. "
        "Discuss limitations. Suggest future directions."
    ),
    "conclusion": (
        "Summarise contributions and findings. No new information. Keep concise."
    ),
    "related work": (
        "Group works by theme, not chronologically. For each group explain what they "
        "do and where they fall short. Use [SOURCE N] extensively."
    ),
    "literature review": (
        "Synthesise by theme. Identify consensus, contradictions, and gaps. "
        "Analyse and compare — do not just list papers. Cite with [SOURCE N]."
    ),
}


def get_section_guidance(instruction: str, document: str) -> str:
    """Detect the paper section and return specific guidance, or empty string."""
    combined = (instruction + " " + document[:300]).lower()
    for section, guidance in SECTION_GUIDANCE.items():
        if section in combined:
            return f"SECTION GUIDANCE ({section.title()}):\n{guidance}\n"
    return ""


# ─────────────────────────────────────────────
# LANGCHAIN PROMPT TEMPLATE
# ─────────────────────────────────────────────

_HUMAN_TEMPLATE = """\
WRITING PREFERENCES:
- Style          : {writing_style}
- Tone           : {tone}
- Language       : {language}
- Citation style : {citation_style}
{journal_line}
{grounding_line}

{section_guidance}
OPERATION: {operation_label}

INSTRUCTION:
{instruction}

{document_block}

{literature_block}

Output ONLY the requested text. No preamble, no explanations.\
"""

# The single prompt template used by agent.py
WRITING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
])


# ─────────────────────────────────────────────
# PROMPT VALUE BUILDER  (called by agent.py)
# ─────────────────────────────────────────────

def build_prompt_values(
    document: str,
    instruction: str,
    context: dict,
    operation: str,
    literature_context: str = "",
) -> dict:
    """
    Build the dict of values to fill into WRITING_PROMPT.

    Args:
        document           : Current section text (empty string = write from scratch).
        instruction        : Researcher's instruction.
        context            : Writing preferences dict from the orchestrator.
        operation          : "generate" | "rephrase" | "improve"
        literature_context : Pre-formatted source chunks string from tools.py.

    Returns:
        Dict — pass directly to  chain.invoke(build_prompt_values(...))
    """

    # Operation label
    labels = {
        "generate": "GENERATE — Write new content from scratch based on the instruction.",
        "rephrase": "REPHRASE — Rewrite the provided text, preserving every original idea.",
        "improve":  "IMPROVE  — Enhance quality, clarity, coherence, and academic level.",
    }
    operation_label = labels.get(operation, labels["generate"])

    # Document block
    if document and document.strip():
        document_block = f'CURRENT TEXT:\n"""\n{document.strip()}\n"""'
    else:
        document_block = "CURRENT TEXT: (empty — write entirely from scratch)"

    # Literature block
    if literature_context and literature_context.strip():
        style = context.get("citation_style", "APA").upper()
        hint = (
            "place [SOURCE N] directly after the claim, e.g.: '...improves accuracy [SOURCE 1].'"
        ) if style == "IEEE" else (
            "place [SOURCE N] after the claim, e.g.: '...as shown in recent work [SOURCE 1].'"
        )
        literature_block = (
            f"LITERATURE SOURCES — cite using [SOURCE N] markers ({hint}):\n"
            f"{literature_context.strip()}"
        )
    else:
        literature_block = "(No literature sources provided — write without citations.)"

    # Optional preference lines
    journal = context.get("target_journal", "")
    journal_line = f"- Target journal : {journal}" if journal else ""
    grounding_line = (
        "- Grounding      : STRICT — only use facts from the provided sources."
        if context.get("grounded_only", True) else ""
    )

    return {
        "writing_style":    context.get("writing_style",  "academic"),
        "tone":             context.get("tone",            "formal"),
        "language":         context.get("language",        "English"),
        "citation_style":   context.get("citation_style",  "APA"),
        "journal_line":     journal_line,
        "grounding_line":   grounding_line,
        "section_guidance": get_section_guidance(instruction, document),
        "operation_label":  operation_label,
        "instruction":      instruction.strip(),
        "document_block":   document_block,
        "literature_block": literature_block,
    }