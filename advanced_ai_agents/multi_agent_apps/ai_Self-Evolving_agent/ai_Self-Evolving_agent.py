# ---- Headless hack: stub tkinter (HITL GUI) ----
import sys, types
if 'tkinter' not in sys.modules:
    sys.modules['tkinter'] = types.ModuleType('tkinter')
    sys.modules['tkinter.ttk'] = types.ModuleType('tkinter.ttk')
    sys.modules['tkinter.filedialog'] = types.ModuleType('tkinter.filedialog')
# ------------------------------------------------

# ---- STUBS for LlamaIndex bits EvoAgentX imports ----
# We create fake modules so EvoAgentX can import them without the real llama-index.
ll_pkg = types.ModuleType("llama_index"); ll_pkg.__path__ = []  # mark as pkg

# subpackages
ll_core = types.ModuleType("llama_index.core")
ll_schema = types.ModuleType("llama_index.core.schema")
ll_core_emb = types.ModuleType("llama_index.core.embeddings")
ll_emb = types.ModuleType("llama_index.embeddings")
ll_az  = types.ModuleType("llama_index.embeddings.azure_openai")

# --- core.schema stubs ---
class _BaseNode:
    def __init__(self, text: str = "", id_: str | None = None, metadata: dict | None = None, **kwargs):
        self.text = text
        self.id_ = id_ or "stub"
        self.metadata = metadata or {}

class TextNode(_BaseNode): pass

class ImageNode(_BaseNode):
    def __init__(self, image: bytes | None = None, **kwargs):
        super().__init__(**kwargs)
        self.image = image

class RelatedNodeInfo:
    def __init__(self, node_id: str | None = None, metadata: dict | None = None, **kwargs):
        self.node_id = node_id or "stub-related"
        self.metadata = metadata or {}

class NodeWithScore:
    def __init__(self, node: object, score: float = 0.0, **kwargs):
        self.node = node
        self.score = score

ll_schema.NodeWithScore = NodeWithScore
ll_schema.TextNode = TextNode
ll_schema.ImageNode = ImageNode
ll_schema.RelatedNodeInfo = RelatedNodeInfo

# --- core.embeddings stubs ---
class BaseEmbedding:
    """Minimal shim to satisfy `from llama_index.core.embeddings import BaseEmbedding`."""
    def __init__(self, *args, **kwargs): pass
    # Optional helpers in case something calls them
    def get_text_embedding(self, text: str):
        return [0.0]
    def get_text_embedding_batch(self, texts):
        return [[0.0] for _ in texts]

ll_core_emb.BaseEmbedding = BaseEmbedding

# --- embeddings.azure_openai stubs ---
class AzureOpenAIEmbedding:
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]

class AzureOpenAIEmbeddingModel:
    text_embedding_3_large = "text-embedding-3-large"
    text_embedding_3_small = "text-embedding-3-small"

ll_az.AzureOpenAIEmbedding = AzureOpenAIEmbedding
ll_az.AzureOpenAIEmbeddingModel = AzureOpenAIEmbeddingModel

# Register in sys.modules
sys.modules['llama_index'] = ll_pkg
sys.modules['llama_index.core'] = ll_core
sys.modules['llama_index.core.schema'] = ll_schema
sys.modules['llama_index.core.embeddings'] = ll_core_emb
sys.modules['llama_index.embeddings'] = ll_emb
sys.modules['llama_index.embeddings.azure_openai'] = ll_az

# Link hierarchy (attribute access)
ll_pkg.core = ll_core
ll_core.schema = ll_schema
ll_core.embeddings = ll_core_emb
ll_pkg.embeddings = ll_emb
ll_emb.azure_openai = ll_az
# ------------------------------------------------------

import os
from dotenv import load_dotenv

# EvoAgentX imports (use OpenAI wrapper pointed to Ollama)
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()  # Loads environment variables from .env file

# === Ollama OpenAI-compatible endpoint ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://217.15.175.196:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
# Force OpenAI SDK/EvoAgentX to talk to Ollama; dummy key is fine
os.environ["OPENAI_BASE_URL"] = OLLAMA_BASE_URL
os.environ["OPENAI_API_KEY"] = os.getenv("OLLAMA_API_KEY", "ollama")

def build_llm(max_tokens=16000, temperature=0.2, stream=True, output_response=True):
    cfg = OpenAILLMConfig(
        model=OLLAMA_MODEL,
        openai_key=os.environ["OPENAI_API_KEY"],
        stream=stream,
        output_response=output_response,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return OpenAILLM(config=cfg)

def main():
    # === LLMs ===
    llm = build_llm(max_tokens=16000, temperature=0.2)
    verification_llm = build_llm(max_tokens=20000, temperature=0.0)

    # === Task goal ===
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_directory = "examples/output/tetris_game"

    # === Build workflow ===
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # optional visualize (no-op in headless)
    workflow_graph.display()

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(
        workflow_graph,
        llm_config=llm.config
    )

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    # === Verify code ===
    code_verifier = CodeVerification()
    output = code_verifier.execute(
        llm=verification_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # === Extract files ===
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
        inputs={"code_string": output, "target_directory": target_directory}
    )

    print(f"Extracted {len(results.extracted_files)} files:")
    for filename, path in results.extracted_files.items():
        print(f"  - {filename}: {path}")

    if results.main_file:
        print(f"\nMain file: {results.main_file}")
        file_type = os.path.splitext(results.main_file)[1].lower()
        if file_type == ".html":
            print("You can open this HTML file in a browser to play the Tetris game")
        else:
            print("This is the main entry point for your application")

if __name__ == "__main__":
    main()
