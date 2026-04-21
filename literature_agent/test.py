from agent import run_literature_agent
import os

# Run the agent (saves files to outputs/ folder)
run_literature_agent(
    project_title="the importance of ai in healthcare",  # used as query and filename
    citation_style="APA",
    use_web_resources=False,
    use_ocr=False
)

# Read and display the generated files
print("\n==================== LITERATURE REVIEW ====================\n")
review_path = "outputs/the importance of ai in healthcare_state_of_the_art.txt"
if os.path.exists(review_path):
    with open(review_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("Review file not found.")

print("\n==================== CITATIONS ====================\n")
citations_path = "outputs/the importance of ai in healthcare_citations.txt"
if os.path.exists(citations_path):
    with open(citations_path, "r", encoding="utf-8") as f:
        print(f.read())
else:
    print("Citations file not found.")