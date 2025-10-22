import os
from dotenv import load_dotenv

# Load environment early
load_dotenv()

# === Force everything to use Ollama ===
API_BASE = os.getenv("OLLAMA_API_BASE", "http://217.15.175.196:11434/v1")
os.environ["OPENAI_API_BASE"] = API_BASE
os.environ["OPENAI_BASE_URL"] = API_BASE

from evoagentx.models import LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

RAW_MODEL = os.getenv("OLLAMA_MODEL", "openai/llama3.2:1b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")

def normalize_model(raw: str) -> str:
    """Normalize model name so it always uses OpenAI-compatible format."""
    raw = (raw or "").strip()
    if not raw:
        return "openai/llama3.2:1b"
    if "/" not in raw:
        return f"openai/{raw}"
    prefix, rest = raw.split("/", 1)
    return f"openai/{rest}" if prefix.lower() != "openai" else raw

def make_ollama_llm(max_tokens=4096):
    model = normalize_model(RAW_MODEL)
    print(f"[LLM] Using base={API_BASE}, model={model}")
    cfg = LiteLLMConfig(
        model=model,
        api_base=API_BASE,
        openai_key=OPENAI_API_KEY,  # dummy
        stream=True,
        output_response=True,
        max_tokens=max_tokens,
    )
    return LiteLLM(config=cfg), cfg

def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_dir = "examples/output/tetris_game"

    llm, llm_cfg = make_ollama_llm(4096)

    wf_gen = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_gen.generate_workflow(goal=goal)
    workflow_graph.display()

    agent_mgr = AgentManager()
    agent_mgr.add_agents_from_workflow(workflow_graph, llm_config=llm_cfg)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_mgr, llm=llm)
    output = workflow.execute()

    verify_llm, _ = make_ollama_llm(6000)
    code_verifier = CodeVerification()
    verified = code_verifier.execute(
        llm=verify_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    os.makedirs(target_dir, exist_ok=True)
    blocks = extract_code_blocks(verified)

    if len(blocks) == 1:
        path = os.path.join(target_dir, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks[0])
        print(f"✅ Done! Open in browser: {path}")
        return

    extractor = CodeExtraction()
    results = extractor.execute(
        llm=llm,
        inputs={"code_string": verified, "target_directory": target_dir}
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for name, path in results.extracted_files.items():
        print(f"  - {name}: {path}")

if __name__ == "__main__":
    main()
