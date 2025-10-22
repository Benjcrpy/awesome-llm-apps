import os
from dotenv import load_dotenv

# Gagamitin lang natin ang LiteLLM para kumonek sa Ollama (OpenAI-compatible endpoint /v1)
from evoagentx.models import LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

# =========================
# 🔑 ENV / CONFIG
# =========================
load_dotenv()

# NOTE: Para sa Ollama OpenAI-compatible API, gumamit ng /v1 sa dulo.
# Hal.: http://217.15.175.196:11434/v1
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://217.15.175.196:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.2:1b")  # LiteLLM format: "ollama/<model_name>"

# =========================
# 🧱 LLM FACTORY (Ollama)
# =========================
def make_ollama_llm(max_tokens: int = 4096):
    """
    Bumubuo ng LLM client na kumakabit sa Ollama (via LiteLLM OpenAI-compatible /v1).
    Walang API key ang kailangan.
    """
    cfg = LiteLLMConfig(
        model=OLLAMA_MODEL,        # e.g., "ollama/llama3.2:1b"
        api_base=OLLAMA_API_BASE,  # e.g., "http://217.15.175.196:11434/v1"
        stream=True,
        output_response=True,
        max_tokens=max_tokens,
        # walang api_key needed for local/remote Ollama
    )
    return LiteLLM(config=cfg), cfg

# ============
# 🚀 MAIN
# ============
def main():
    # 🎯 Goal (pwede mong palitan later)
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"

    # 1) Planner/Executor: parehong Ollama (llama3.2:1b)
    llm, llm_cfg = make_ollama_llm(max_tokens=4096)
    wf_gen = WorkFlowGenerator(llm=llm)

    # 2) PLAN
    workflow_graph: WorkFlowGraph = wf_gen.generate_workflow(goal=goal)
    workflow_graph.display()

    # 3) BUILD agents
    agent_mgr = AgentManager()
    agent_mgr.add_agents_from_workflow(workflow_graph, llm_config=llm_cfg)

    # 4) EXECUTE
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_mgr, llm=llm)
    output = workflow.execute()

    # 5) VERIFY — gamitin din ang Ollama (same model) para walang external keys
    verify_llm, verify_cfg = make_ollama_llm(max_tokens=6000)
    code_verifier = CodeVerification()
    verified = code_verifier.execute(
        llm=verify_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # 6) EXTRACT → files
    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)

    if len(blocks) == 1:
        path = os.path.join(target_dir, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks[0])
        print(f"✅ Tapos! Buksan mo sa browser: {path}")
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
            print("✅ Buksan mo itong HTML file sa browser para laruin ang Tetris.")
        else:
            print("ℹ️ Ito ang main entry point ng app.")

if __name__ == "__main__":
    main()
