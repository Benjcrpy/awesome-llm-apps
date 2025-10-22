import os
from dotenv import load_dotenv
from evoagentx.models import (
    OpenAILLMConfig, OpenAILLM,
    LiteLLMConfig, LiteLLM
)
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

def make_openai_llm():
    cfg = OpenAILLMConfig(
        model="gpt-4o-mini",
        openai_key=OPENAI_API_KEY,
        stream=True,
        output_response=True,
        max_tokens=8000,
    )
    return OpenAILLM(config=cfg), cfg

def make_claude_llm():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Walang ANTHROPIC_API_KEY para sa fallback.")
    cfg = LiteLLMConfig(
        model="anthropic/claude-3-7-sonnet-20250219",
        anthropic_key=ANTHROPIC_API_KEY,
        stream=True,
        output_response=True,
        max_tokens=10000,
    )
    return LiteLLM(config=cfg), cfg

def plan_with_fallback(goal):
    llm, cfg = make_openai_llm()
    gen = WorkFlowGenerator(llm=llm)
    try:
        graph = gen.generate_workflow(goal=goal)
        return graph, llm, cfg
    except Exception as e:
        if "429" in str(e) or "insufficient_quota" in str(e):
            print("⚠️ Naubos quota sa OpenAI, lilipat tayo sa Claude...")
            fb_llm, fb_cfg = make_claude_llm()
            fb_gen = WorkFlowGenerator(llm=fb_llm)
            graph = fb_gen.generate_workflow(goal=goal)
            return graph, fb_llm, fb_cfg
        raise e

def main():
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target = "examples/output/tetris_game"
    graph, llm, cfg = plan_with_fallback(goal)

    graph.display()
    manager = AgentManager()
    manager.add_agents_from_workflow(graph, llm_config=cfg)
    workflow = WorkFlow(graph=graph, agent_manager=manager, llm=llm)
    output = workflow.execute()

    if not ANTHROPIC_API_KEY:
        verified = output
    else:
        verify_cfg = LiteLLMConfig(
            model="anthropic/claude-3-7-sonnet-20250219",
            anthropic_key=ANTHROPIC_API_KEY,
            stream=True,
            output_response=True,
            max_tokens=20000,
        )
        verify_llm = LiteLLM(config=verify_cfg)
        verify = CodeVerification()
        verified = verify.execute(
            llm=verify_llm, inputs={"requirements": goal, "code": output}
        ).verified_code

    os.makedirs(target, exist_ok=True)
    blocks = extract_code_blocks(verified)

    if len(blocks) == 1:
        path = os.path.join(target, "index.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks[0])
        print(f"✅ Tapos! Buksan sa browser: {path}")
        return

    extractor = CodeExtraction()
    result = extractor.execute(llm=llm, inputs={"code_string": verified, "target_directory": target})
    print(f"Extracted {len(result.extracted_files)} files:")
    for name, path in result.extracted_files.items():
        print(f"  - {name}: {path}")
    if result.main_file:
        print(f"\nMain file: {result.main_file}")

if __name__ == "__main__":
    main()
