import os
from dotenv import load_dotenv

# Load env early
load_dotenv()

from evoagentx.models import LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

# ---------- ENV ----------
API_BASE = os.getenv("OLLAMA_API_BASE", "http://217.15.175.196:11434/v1")
RAW_MODEL = os.getenv("OLLAMA_MODEL", "custom_openai/llama3.2:1b")
DUMMY_KEY = os.getenv("OPENAI_API_KEY", "dummy")  # required field but unused by Ollama

def normalize_to_custom_openai(raw: str) -> str:
    """
    Always return a 'custom_openai/<model>' string.
    Accepts:
      - 'custom_openai/llama3.2:1b'  -> kept
      - 'openai/llama3.2:1b'        -> 'custom_openai/llama3.2:1b'
      - 'ollama/llama3.2:1b'        -> 'custom_openai/llama3.2:1b'
      - 'llama3.2:1b'               -> 'custom_openai/llama3.2:1b'
    """
    raw = (raw or "").strip()
    if not raw:
        return "custom_openai/llama3.2:1b"
    if "/" not in raw:
        return f"custom_openai/{raw}"
    prefix, rest = raw.split("/", 1)
    return f"custom_openai/{rest}" if prefix.lower() != "custom_openai" else raw

def make_ollama_llm(max_tokens: int = 4096):
    model = normalize_to_custom_openai(RAW_MODEL)
    # Print to logs for verification
    print(f"[LLM] provider=custom_openai  api_base={API_BASE}  model={model}")
    cfg = LiteLLMConfig(
        model=model,           # e.g., 'custom_openai/llama3.2:1b'
        api_base=API_BASE,     # Ollama OpenAI-compatible endpoint
        openai_key=DUMMY_KEY,  # dummy header; Ollama ignores it
        stream=True,
        output_response=True,
        max_tokens=max_tokens,
    )
    return LiteLLM(config=cfg), cfg

def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"

    # Use Ollama (planning & execution)
    llm, llm_cfg = make_ollama_llm(4096)

    # Plan
    wf_gen = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_gen.generate_workflow(goal=goal)
    workflow_graph.display()

    # Agents
    agent_mgr = AgentManager()
    agent_mgr.add_agents_from_workflow(workflow_graph, llm_config=llm_cfg)

    # Execute
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_mgr, llm=llm)
    output = workflow.execute()

    # Verify using the same Ollama-backed client
    verify_llm, _ = make_ollama_llm(6000)
    code_verifier = CodeVerification()
    verified = code_verifier.execute(
        llm=verify_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # Extract to files
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
    main()
