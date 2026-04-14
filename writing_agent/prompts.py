"""
prompts.py — Writing Agent Prompt Templates (LangChain)
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT  (writing agent)
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
9. When writing a named section (abstract, introduction, methodology …), follow the
   structural blueprint provided for that section exactly. Do not skip sub-parts.
"""

# ─────────────────────────────────────────────
# SYSTEM PROMPT  (suggestion agent — kept short
#                so the LLM stays focused)
# ─────────────────────────────────────────────

SUGGESTION_SYSTEM_PROMPT = """You are a real-time inline writing assistant for CoWriteX, \
an academic paper writing tool. Your job is to generate a single, concise AI suggestion \
that the researcher can either accept or reject with one keystroke.

STRICT RULES:
1. Return ONLY a JSON object — no prose, no markdown fences.
2. The JSON must have exactly two keys:
   - "original": the exact text you were asked to improve/rephrase/complete
                 (empty string if this is a fresh completion at cursor).
   - "suggestion": your improved or completed text, same approximate length.
3. Do not introduce new claims unsupported by the provided context.
4. Never explain what you changed. Never apologise. Output JSON only.
"""

# ─────────────────────────────────────────────
# SECTION STRUCTURE BLUEPRINTS
# ─────────────────────────────────────────────
# Each value is a tuple of (structural_blueprint, style_notes).
# The blueprint is injected verbatim into the prompt so the LLM
# must follow the numbered outline.

SECTION_BLUEPRINTS: dict[str, tuple[str, str]] = {
    "abstract": (
        """Required structure (write in this order, no sub-headings):
  1. Background / motivation       — 1–2 sentences: what problem exists and why it matters.
  2. Research gap                  — 1 sentence: what is missing in current knowledge.
  3. Objective / approach          — 1–2 sentences: what this paper does and how.
  4. Key results                   — 1–2 sentences: most important quantitative or qualitative findings.
  5. Significance / conclusion     — 1 sentence: broader impact or takeaway.
Total target: 150–250 words. Past tense for methods/results, present tense for conclusions.""",
        "No sub-headings. Dense, single block of prose.",
    ),

    "introduction": (
        """Required structure (write each part as a distinct paragraph):
  § 1  Context & motivation        — Establish the domain; explain why this research area matters.
  § 2  Related work & gap          — Briefly survey key prior work; identify the gap or limitation \
your paper addresses. Use [SOURCE N] markers.
  § 3  Contribution statement      — State your specific contributions clearly (can use a short \
list if 3+ contributions).
  § 4  Paper organisation          — One short paragraph: "The rest of this paper is organised as \
follows: Section 2 covers …"
Target: 500–800 words total, 4 paragraphs minimum.""",
        "Funnel structure: broad → specific. Each paragraph flows into the next.",
    ),

    "related work": (
        """Required structure:
  § Opening                        — 2–3 sentences framing which research threads are covered.
  § Thematic group 1               — Name the theme. Discuss 2–4 papers. End with what they lack. \
Use [SOURCE N].
  § Thematic group 2               — Same pattern for a second theme.
  § (Optional) Thematic group 3    — Third theme if relevant.
  § Positioning paragraph          — How your work differs from / builds on everything above.
Group papers by theme, NOT chronologically. Cite with [SOURCE N] extensively.""",
        "Analytical synthesis. Never just list papers. Always end each group with a gap statement.",
    ),

    "literature review": (
        """Required structure:
  § Opening paragraph              — Scope and organisation of the review.
  § Theme A                        — Synthesise 3+ works: consensus, contradictions, progress.
  § Theme B                        — Same for a second cluster.
  § Theme C  (if applicable)       — Third cluster.
  § Critical analysis paragraph    — Identify research gaps across all themes.
  § Summary paragraph              — Key insights that motivate your research.
Cite with [SOURCE N] throughout. Do not just summarise — compare, contrast, and critique.""",
        "This is a review section; depth and critical analysis are mandatory.",
    ),

    "methodology": (
        """Required structure (separate paragraphs, no sub-headings unless journal requires):
  § Dataset / experimental setup   — Source, size, splits, preprocessing steps.
  § Model / algorithm description  — Architecture, key hyper-parameters, design choices and why.
  § Training / optimisation        — Loss function, optimiser, hardware, duration.
  § Evaluation protocol            — Metrics, baselines, statistical tests used.
Passive voice is standard in formal journals. Be precise enough for full reproducibility.""",
        "Precision and reproducibility are the primary goals.",
    ),

    "results": (
        """Required structure:
  § Overview sentence              — 1 sentence summarising the overall outcome.
  § Main quantitative results      — Report numbers with units; refer to tables/figures \
(e.g. "Table 1 shows …").
  § Comparison to baselines        — How does your approach compare? Use concrete deltas.
  § Ablation / secondary findings  — If applicable, note what each component contributes.
Do NOT interpret results here — save that for the Discussion section.
Use past tense throughout.""",
        "Objective reporting only. No interpretation or speculation.",
    ),

    "discussion": (
        """Required structure:
  § Interpretation of results      — What do the results mean relative to the research question?
  § Comparison with prior work     — How do results align with or contradict [SOURCE N]?
  § Limitations                    — Be honest about scope, data, or method constraints.
  § Future work                    — Concrete directions that would address the limitations.
Present tense for general claims, past tense when referring to specific experiments.""",
        "Intellectual depth expected. Limitations must not be omitted.",
    ),

    "conclusion": (
        """Required structure:
  § Summary of contributions       — 2–3 sentences re-stating what was done and found.
  § Significance                   — Why does it matter? Broader implications.
  § Future directions              — 1–2 sentences only; do not open new topics.
No new information. Keep under 200 words. Present tense throughout.""",
        "Concise closure. Mirror the introduction's contribution statement.",
    ),
}


def get_section_blueprint(instruction: str, document: str) -> tuple[str, str]:
    """
    Detect the target section from the instruction/document text and
    return (blueprint_text, style_notes).
    Returns ("", "") if no section is detected.
    """
    combined = (instruction + " " + document[:300]).lower()
    for section_key, (blueprint, style) in SECTION_BLUEPRINTS.items():
        if section_key in combined:
            return blueprint, style
    return "", ""


# ─────────────────────────────────────────────
# WRITING AGENT — LANGCHAIN PROMPT TEMPLATE
# ─────────────────────────────────────────────

_HUMAN_TEMPLATE = """\
WRITING PREFERENCES:
- Style          : {writing_style}
- Tone           : {tone}
- Language       : {language}
- Citation style : {citation_style}
{journal_line}
{grounding_line}

OPERATION: {operation_label}

{section_block}

INSTRUCTION:
{instruction}

{document_block}

{literature_block}

Output ONLY the requested text. No preamble, no explanations.\
"""

WRITING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(_HUMAN_TEMPLATE),
])


# ─────────────────────────────────────────────
# SUGGESTION AGENT — LANGCHAIN PROMPT TEMPLATE
# ─────────────────────────────────────────────

_SUGGESTION_HUMAN_TEMPLATE = """\
WRITING PREFERENCES:
- Style   : {writing_style}
- Tone    : {tone}
- Language: {language}
{journal_line}

SUGGESTION MODE: {suggestion_mode}
{section_hint}

FULL DOCUMENT CONTEXT (for coherence — do NOT reproduce verbatim):
\"\"\"
{document_context}
\"\"\"

TARGET TEXT TO WORK ON:
\"\"\"
{target_text}
\"\"\"

{literature_block}

Return ONLY a valid JSON object with keys "original" and "suggestion".\
"""

SUGGESTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SUGGESTION_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(_SUGGESTION_HUMAN_TEMPLATE),
])


# ─────────────────────────────────────────────
# PROMPT VALUE BUILDERS
# ─────────────────────────────────────────────

def build_prompt_values(
    document: str,
    instruction: str,
    context: dict,
    operation: str,
    literature_context: str = "",
) -> dict:
    """Build the dict of values for WRITING_PROMPT."""

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
            "place [SOURCE N] directly after the claim"
            if style == "IEEE"
            else "place [SOURCE N] after the claim"
        )
        literature_block = (
            f"LITERATURE SOURCES — cite using [SOURCE N] markers ({hint}):\n"
            f"{literature_context.strip()}"
        )
    else:
        literature_block = "(No literature sources provided — write without citations.)"

    # Section blueprint
    blueprint, style_notes = get_section_blueprint(instruction, document)
    if blueprint:
        section_block = (
            f"SECTION STRUCTURE — follow this blueprint exactly:\n{blueprint}\n"
            + (f"\nSTYLE NOTE: {style_notes}" if style_notes else "")
        )
    else:
        section_block = ""

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
        "section_block":    section_block,
        "operation_label":  operation_label,
        "instruction":      instruction.strip(),
        "document_block":   document_block,
        "literature_block": literature_block,
    }


def build_suggestion_prompt_values(
    document: str,
    target_text: str,
    context: dict,
    suggestion_mode: str,
    literature_context: str = "",
) -> dict:
    """
    Build the dict of values for SUGGESTION_PROMPT.

    Args:
        document        : Full document so far (for coherence context).
        target_text     : The specific text to complete / improve / rephrase.
                          Empty string means inline completion at cursor.
        context         : Writing preferences dict.
        suggestion_mode : "complete" | "improve" | "rephrase"
        literature_context : Pre-formatted source string (optional).

    Returns:
        Dict — pass directly to suggestion_chain.invoke(...)
    """
    mode_descriptions = {
        "complete":  "COMPLETE — continue the sentence or paragraph naturally from where it ends.",
        "improve":   "IMPROVE  — enhance academic quality, clarity, and precision of the target text.",
        "rephrase":  "REPHRASE — rewrite the target text in a different way, keeping the meaning.",
    }
    mode_label = mode_descriptions.get(suggestion_mode, mode_descriptions["improve"])

    # Detect section for a helpful hint
    blueprint, _ = get_section_blueprint(document[:500], "")
    section_hint = (
        f"SECTION CONTEXT: This text appears in the {_detect_section_from_doc(document)} section.\n"
        if _detect_section_from_doc(document)
        else ""
    )

    if literature_context and literature_context.strip():
        lit_block = f"RELEVANT SOURCES (use for grounding if applicable):\n{literature_context.strip()}"
    else:
        lit_block = ""

    journal = context.get("target_journal", "")

    return {
        "writing_style":     context.get("writing_style", "academic"),
        "tone":              context.get("tone",           "formal"),
        "language":          context.get("language",       "English"),
        "journal_line":      f"- Target journal: {journal}" if journal else "",
        "suggestion_mode":   mode_label,
        "section_hint":      section_hint,
        "document_context":  document[:1500] if document else "(empty document)",
        "target_text":       target_text if target_text else "(write a continuation from the document above)",
        "literature_block":  lit_block,
    }


def _detect_section_from_doc(document: str) -> str:
    """Light heuristic to name which section the document belongs to."""
    lower = document[:600].lower()
    for section in SECTION_BLUEPRINTS:
        if section in lower:
            return section
    return ""