import os
from dotenv import load_dotenv

from evoagentx.models import LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

# ===============================
# ENV SETUP
# ===============================
load_dotenv()

API_BASE = os.getenv("OLLAMA_API_BASE", "http://217.15.175.196:11434/v1")
RAW_MODEL = os.getenv("OLLAMA_MODEL", "openai/llama3.2:1b")   # will be normalized
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")    # dummy key, required by 'openai' provider

def _normalize_openai_model(raw: str) -> str:
    """
    Accepts:
      - "openai/llama3.2:1b"  (already correct)
      - "ollama/llama3.2:1b"  -> becomes "openai/llama3.2:1b"
      - "llama3.2:1b"         -> becomes "openai/llama3.2:1b"
    """
    raw = (raw or "").strip()
    if not raw:
        return "openai/llama3.2:1b"
    if "/" not in raw:
        return f"openai/{raw}"
    prefix, rest = raw.split("/", 1)
    return f"openai/{rest}" if prefix.lower() != "openai" else raw

# ===============================
# LLM Factory (Ollama via OpenAI-compatible /v1)
# ===============================
def make_ollama_llm(max_tokens: int = 4096):
    model = _normalize_openai_model(RAW_MODEL)
    print(f"[LLM] api_base={API_BASE}  model={model}")  # visible in Coolify logs
    cfg = LiteLLMConfig(
        model=model,               # e.g., "openai/llama3.2:1b"
        api_base=API_BASE,         # e.g., "http://217.15.175.196:11434/v1"
        openai_key=OPENAI_API_KEY, # dummy OK for 'openai' provider
        stream=True,
        output_response=True,
        max_tokens=max_tokens,
    )
    return LiteLLM(config=cfg), cfg

# ===============================
# MAIN
# ===============================
def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"

    # 1) Build LLM (planner & executor both use Ollama)
    llm, llm_cfg = make_ollama_llm(max_tokens=4096)

    # 2) Plan
    wf_gen = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_gen.generate_workflow(goal=goal)
    workflow_graph.display()

    # 3) Agents
    agent_mgr = AgentManager()
    agent_mgr.add_agents_from_workflow(workflow_graph, llm_config=llm_cfg)

    # 4) Execute
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_mgr, llm=llm)
    output = workflow.execute()

    # 5) Verify (also via Ollama)
    verify_llm, _ = make_ollama_llm(max_tokens=6000)
    code_verifier = CodeVerification()
    verified = code_verifier.execute(
        llm=verify_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # 6) Extract files
    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)

    if len(blocks) == 1:
        file_path = os.path.join(target_dir, "index.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(blocks[0])
        print(f"✅ Done! Open in your browser: {file_path}")
        return

    extractor = CodeExtraction()
    results = extractor.execute(
        llm=llm,
        inputs={"code_string": verified, "target_directory": target_dir}
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for name, path in results.extracted_files.items():
        print(f"  - {name}: {path}")

    if results.main_file:
        print(f"\nMain file: {results.main_file}")
        ext = os.path.splitext(results.main_file)[1].lower()
        if ext == ".html":
            print("✅ Open this HTML file in your browser to play Tetris.")
        else:
            print("ℹ️ This is the main entry point of your app.")

if __name__ == "__main__":
    main()
