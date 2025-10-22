import os
from dotenv import load_dotenv

# Load env early
load_dotenv()

# ===== Force OpenAI client to point to OLLAMA =====
API_BASE = os.getenv("OLLAMA_API_BASE", "http://217.15.175.196:11434/v1")
os.environ["OPENAI_BASE_URL"] = API_BASE       # new OpenAI SDK env
os.environ["OPENAI_API_BASE"] = API_BASE       # some wrappers read this name too
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy")  # Ollama doesn't check it
# ==================================================

from evoagentx.models import (
    OpenAILLMConfig, OpenAILLM,   # use OpenAI client but base_url = Ollama
)
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

# Model name that exists in your Ollama server
RAW_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")  # no provider prefix here

def make_ollama_openai_llm(max_tokens: int = 4096):
    """
    Use EvoAgentX's OpenAI wrapper but force base_url to your Ollama /v1 endpoint.
    """
    cfg = OpenAILLMConfig(
        model=RAW_MODEL,                 # e.g., "llama3.2:1b"
        openai_key=os.environ["OPENAI_API_KEY"],  # any dummy value
        stream=True,
        output_response=True,
        max_tokens=max_tokens,
        # IMPORTANT: EvoAgentX reads base URL from env; we already set it above.
        # If your EvoAgentX version supports it, also pass api_base explicitly:
        api_base=API_BASE,               # <-- double insurance
    )
    return OpenAILLM(config=cfg), cfg

def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"

    # Planner + Executor (both via Ollama)
    llm, llm_cfg = make_ollama_openai_llm(4096)

    wf_gen = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_gen.generate_workflow(goal=goal)
    workflow_graph.display()

    agent_mgr = AgentManager()
    agent_mgr.add_agents_from_workflow(workflow_graph, llm_config=llm_cfg)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_mgr, llm=llm)
    output = workflow.execute()

    # Verify using the same client (Ollama)
    verify_llm, _ = make_ollama_openai_llm(6000)
    code_verifier = CodeVerification()
    verified = code_verifier.execute(
        llm=verify_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # Extract files
    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)

    if len(blocks) == 1:
        path = os.path.join(target_dir, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks[0])
        print(f"Done! Open in browser: {path}")
        return

    extractor = CodeExtraction()
    results = extractor.execute(
        llm=llm,
        inputs={"code_string": verified, "target_directory": target_dir}
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for name, p in results.extracted_files.items():
        print(f"  - {name}: {p}")

if __name__ == "__main__":
    # Debug prints to be sure routing is correct
    print(f"[BOOT] OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL')}")
    print(f"[BOOT] OPENAI_API_BASE={os.environ.get('OPENAI_API_BASE')}")
    print(f"[BOOT] MODEL={RAW_MODEL}")
    main()
