"""
pdf_to_mermaid.py
=================
Pipeline to parse a PDF file, extract and clean the text, and use the Grok API
to generate a valid Mermaid diagram representation.
"""

import os
import requests
import fitz  # PyMuPDF
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # This tells Python to actually read from the .env file

class GrokMermaidGenerator:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        # The API key can be passed directly or picked up from the environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from the given PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")
        return text

    def clean_text(self, raw_text: str, max_chars: int = 10000) -> str:
        """Clean and select a segment of the text to fit within context limits."""
        # Clean up whitespace and newlines for an easier read by the API
        cleaned = " ".join(raw_text.split())
        
        # Take the most relevant initial segment to keep the diagram focused
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
        return cleaned

    def generate_diagram(self, text_chunk: str) -> str:
        """Send the text to Grok API to generate a Mermaid diagram."""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set the environment variable.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Strictly define the persona and negative constraints as requested
        system_prompt = (
            "You are an expert in generating Mermaid.js diagrams for a visualization platform.\n"
            "Your task is to convert a natural language description into a valid Mermaid diagram.\n\n"
            "## DIAGRAM TYPE SELECTION\n"
            "Automatically choose the most appropriate diagram type:\n"
            "1. FLOWCHART 2. SEQUENCE DIAGRAM 3. ER DIAGRAM 4. STATE DIAGRAM\n"
            "5. GANTT CHART 6. CLASS DIAGRAM 7. MINDMAP 8. PIE CHART\n\n"
            "## OUTPUT RULES (STRICT)\n"
            "- Output ONLY Mermaid syntax\n"
            "- DO NOT include explanations, comments, or text outside the diagram\n"
            "- DO NOT wrap output in markdown (no ``` blocks)\n"
            "- Keep diagram clean and readable\n"
            "- Limit to 6-10 nodes/elements maximum\n"
            "- Use simple labels (short and clear)\n"
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a Mermaid diagram summarizing the following text:\n\n{text_chunk}"}
            ],
            "temperature": 0.1
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Grok API error: {response.status_code} - {response.text}")
            
        data = response.json()
        mermaid_syntax = data["choices"][0]["message"]["content"].strip()
        
        # Extra safety cleanup incase Grok wraps it despite instructions
        if mermaid_syntax.startswith("```mermaid"):
            mermaid_syntax = mermaid_syntax[len("```mermaid"):].strip()
        if mermaid_syntax.startswith("```"):
            mermaid_syntax = mermaid_syntax[len("```"):].strip()
        if mermaid_syntax.endswith("```"):
            mermaid_syntax = mermaid_syntax[:-len("```")].strip()
            
        # Fix hallucinated invalid transitions (e.g. `-->|distribute|>` instead of `-->|distribute|`)
        import re
        mermaid_syntax = re.sub(r'-->\|([^|]+)\|>', r'-->|\1|', mermaid_syntax)
            
        return mermaid_syntax

    def process_pdf(self, pdf_path: str) -> str:
        """End-to-end pipeline: PDF -> Text -> Grok API -> Mermaid Syntax"""
        raw_text = self.extract_text(pdf_path)
        clean_chunk = self.clean_text(raw_text)
        mermaid_syntax = self.generate_diagram(clean_chunk)
        return mermaid_syntax

    def save_as_image(self, mermaid_syntax: str, filename: str = "mermaid_chart", fmt: str = "png") -> str:
        """Converts mermaid syntax to PNG or PDF image using mermaid.ink API and saves to outputs folder.

        Parameters
        ----------
        mermaid_syntax : str  — the raw Mermaid code
        filename       : str  — output filename stem (no extension)
        fmt            : str  — "png" (default) or "pdf"
        """
        encoded_str = base64.urlsafe_b64encode(mermaid_syntax.encode('utf-8')).decode('utf-8')

        if fmt == "pdf":
            url = f"https://mermaid.ink/pdf/{encoded_str}"
        else:
            fmt = "png"  # default fallback
            url = f"https://mermaid.ink/img/{encoded_str}"

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate {fmt.upper()} via Mermaid.ink: status {response.status_code}"
            )

        OUTPUTS_DIR = Path(__file__).parent / "outputs"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        out_path = OUTPUTS_DIR / f"{stem}.{fmt}"

        with open(out_path, "wb") as f:
            f.write(response.content)

        return str(out_path.resolve())

    def convert_mermaid_to_tikz(self, mermaid_syntax: str) -> str:
        """Uses the Groq API to convert Mermaid syntax into pure native TikZ LaTeX code."""
        system_prompt = (
            "You are an expert LaTeX and TikZ programmer.\n"
            "Your task is to convert the provided Mermaid diagram syntax into a pure, standalone LaTeX document "
            "using the TikZ package.\n\n"
            "## OUTPUT RULES (STRICT)\n"
            "- Output ONLY valid LaTeX code.\n"
            "- Do NOT include markdown blocks like ```latex.\n"
            "- The output MUST be a full compiling document starting with \\documentclass{standalone}.\n"
            "- It MUST use \\usepackage{tikz} and \\usetikzlibrary{shapes,arrows.meta,positioning}.\n"
            "- **CRITICAL LAYOUT & SIZING RULES**:\n"
            "  1. Dynamically analyze the graph: choose a robust `node distance` so that NO text/labels overlap with boxes.\n"
            "  2. If the graph has many nodes (>5), use `font=\\scriptsize` or `\\tiny` inside the actual nodes to keep the diagram compact and legible.\n"
            "  3. Use `font=\\tiny` for all edge/arrow labels.\n"
            "  4. Ensure the structural layout folds cleanly (using `below right`, `below`, `left`) so the final graphic comfortably approximates standard physical dimensions without infinitely stretching horizontally or squashing itself.\n"
            "  5. Add `align=center` to all nodes so their text can wrap securely.\n"
            "- DO NOT provide any conversational text before or after the code."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this Mermaid diagram to pure TikZ LaTeX:\n\n{mermaid_syntax}"}
            ],
            "temperature": 0.1
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Grok API error during TikZ conversion: {response.status_code} - {response.text}")
            
        data = response.json()
        tikz_code = data["choices"][0]["message"]["content"].strip()
        
        # Cleanup any accidental markdown block wrappers
        if tikz_code.startswith("```latex"):
            tikz_code = tikz_code[len("```latex"):].strip()
        if tikz_code.startswith("```"):
            tikz_code = tikz_code[len("```"):].strip()
        if tikz_code.endswith("```"):
            tikz_code = tikz_code[:-len("```")].strip()
            
        return tikz_code

    def save_as_latex(self, mermaid_syntax: str, filename: str = "mermaid_chart", title: str = "Generated Diagram") -> str:
        """Exports the Mermaid diagram as a standalone pure TikZ LaTeX .tex file.

        Instead of outputting an image or raw text, this dynamically translates the 
        Mermaid syntax into pure, native TikZ code using the LLM, ensuring zero 
        external file dependencies (no .png or .pdf graphics needed).
        """
        # 1. Ask LLM to translate mermaid to tikz
        print("[**] Translating Mermaid to native TikZ via LLM...")
        tikz_code = self.convert_mermaid_to_tikz(mermaid_syntax)

        OUTPUTS_DIR = Path(__file__).parent / "outputs"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        out_path = OUTPUTS_DIR / f"{stem}.tex"

        # 2. Write the pure TikZ LaTeX output
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tikz_code)

        return str(out_path.resolve())

if __name__ == "__main__":
    # --- HARDCODE YOUR PATH HERE ---
    pdf_file = "bayern_vs_psg_attack_flow.pdf"
    out_name  = "bayern_vs_psg_attack_flow"    # stem used for all output files
    # --------------------------------

    try:
        generator = GrokMermaidGenerator()

        # 1. Generate Mermaid syntax from PDF
        mermaid_output = generator.process_pdf(pdf_file)
        print("\n\n======== FINAL MERMAID ========\n")
        print(mermaid_output)

        # 2. Export as PNG
        print("\n\n======== EXPORT: PNG ========\n")
        png_path = generator.save_as_image(mermaid_output, filename=out_name, fmt="png")
        print(f"PNG  saved -> {png_path}")

        # 3. Export as PDF
        print("\n\n======== EXPORT: PDF ========\n")
        pdf_path = generator.save_as_image(mermaid_output, filename=out_name, fmt="pdf")
        print(f"PDF  saved -> {pdf_path}")

        # 4. Export as LaTeX
        print("\n\n======== EXPORT: LaTeX ========\n")
        tex_path = generator.save_as_latex(mermaid_output, filename=out_name, title="Generated Diagram")
        print(f"LaTeX saved -> {tex_path}")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

