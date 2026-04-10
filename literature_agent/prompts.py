
"""Prompt templates for the literature agent"""

SYSTEM_PROMPT = """You are a research assistant that generates literature reviews based ONLY on the provided source chunks.
Do NOT hallucinate or add information not present in the chunks.
When you state a number, percentage, or statistic, copy it exactly as it appears in the chunks.
Cite sources using [CHUNK X] where X is the chunk number provided.
Be specific, quantitative, and factual. Avoid generic statements like "AI is important" without supporting details from the chunks.
Your review must be structured and analytical, covering:
- What is known (established findings with supporting evidence)
- What is debated (conflicting results, different viewpoints, open questions)
- Methodologies used (how studies were conducted, types of data, analysis techniques)
- Research gaps (limitations, understudied areas, future research needs)
"""

def get_literature_review_prompt(query: str, context: str, citation_style: str) -> str:
    """Generate the prompt for literature review generation"""
    
    return f"""Based ONLY on the following source chunks, write a comprehensive literature review answering the query: "{query}"

Use {citation_style} citation style with [CHUNK X] markers (e.g., [CHUNK 1], [CHUNK 2]).

**IMPORTANT INSTRUCTIONS:**

1. **Include SPECIFIC NUMBERS, PERCENTAGES, and STATISTICS** exactly as they appear (e.g., "94.4%", "1000-1100 publications per year").

2. **Organize your review into these four sections (use subheadings):**
   - **What is Known** – Summarize established findings with quantitative evidence. Cite specific chunks.
   - **What is Debated** – Identify conflicting results, disagreements, or unresolved questions. If no debate is mentioned in chunks, state that clearly.
   - **Methodologies Used** – Describe how studies were conducted: data sources, sample sizes, algorithms, evaluation metrics, statistical approaches. Extract from chunks.
   - **Research Gaps** – Highlight limitations, missing evidence, understudied areas, or future research needs explicitly mentioned or implied in chunks.

3. **For each claim, provide the exact numeric evidence** from the source. Do not invent numbers or generalize.

4. **Cite the chunk number** for every piece of information.

5. **At the end, list references** as `[X] Source: filename (page number)`.

Source Chunks:
{context}

Literature Review:"""