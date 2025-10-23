import os
from dotenv import load_dotenv

# EvoAgentX imports (unchanged logic)
from evoagentx.models import OpenAILLMConfig, OpenAILLM, LiteLLMConfig, LiteLLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()  # Loads environment variables from .env file

# === Keys / URLs ===
# We will route ALL LLM calls to Ollama via OpenAI-compatible endpoint.
# Keep Anthropic unset; we won't call it now.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://217.15.175.196:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # dummy token for Ollama

def build_generation_llm():
    """
    Main LLM used across:
      - WorkFlowGenerator
      - Agents created by AgentManager
      - WorkFlow.execute()

    We switch from OpenAILLM -> LiteLLM to use Ollama's OpenAI-compatible API.
    """
    gen_cfg = LiteLLMConfig(
        # IMPORTANT: use litellm's "ollama/<model>" routing with explicit base_url
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        # Parity with original:
        stream=True,
        output_response=True,
        max_tokens=16000,
        temperature=0.2,  # optional, stable gen
    )
    return LiteLLM(config=gen_cfg)

def build_verifier_llm():
    """
    Previously used Anthropic via LiteLLM.
    We'll reuse Ollama model for verification to keep the flow intact.
    """
    ver_cfg = LiteLLMConfig(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        stream=True,
        output_response=True,
        max_tokens=20000,
        temperature=0.0,  # stricter for verification
    )
    return LiteLLM(config=ver_cfg)

def main():
    # === LLMs ===
    llm = build_generation_llm()

    # === Your goal (unchanged) ===
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_directory = "examples/output/tetris_game"

    # === Build workflow (unchanged logic) ===
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # [optional] display/save
    workflow_graph.display()
    # workflow_graph.save_module(f"{target_directory}/workflow_demo_ollama.json")
    # workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(f"{target_directory}/workflow_demo_ollama.json")

    agent_manager = AgentManager()
    # NOTE: pass the SAME llm config used above so agents are consistent
    agent_manager.add_agents_from_workflow(
        workflow_graph,
        llm_config=llm.config  # keep agents on the Ollama LiteLLM config
    )

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    # === Verification (reuse Ollama) ===
    verification_llm = build_verifier_llm()
    code_verifier = CodeVerification()
    output = code_verifier.execute(
        llm=verification_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # === Extraction (unchanged) ===
    os.makedirs(target_directory, exist_ok=True)
    code_blocks = extract_code_blocks(output)
    if len(code_blocks) == 1:
        file_path = os.path.join(target_directory, "index.html")
        with open(file_path, "w") as f:
            f.write(code_blocks[0])
        print(f"You can open this HTML file in a browser to play the Tetris game: {file_path}")
        return

    code_extractor = CodeExtraction()
    results = code_extractor.execute(
        llm=llm,
        inputs={
            "code_string": output,
            "target_directory": target_directory,
        }
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for filename, path in results.extracted_files.items():
        print(f"  - {filename}: {path}")

    if results.main_file:
        print(f"\nMain file: {results.main_file}")
        file_type = os.path.splitext(results.main_file)[1].lower()
        if file_type == '.html':
            print(f"You can open this HTML file in a browser to play the Tetris game")
        else:
            print(f"This is the main entry point for your application")

if __name__ == "__main__":
    main()
