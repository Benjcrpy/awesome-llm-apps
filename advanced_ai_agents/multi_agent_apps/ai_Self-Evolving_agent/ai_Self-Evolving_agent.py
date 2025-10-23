# --- HARD STUB: dashscope (to satisfy evoagentx.models.aliyun_model import) ---
import sys, types
try:
    from dashscope import Generation  # if the real SDK exists, ok
except Exception:
    class _DummyGeneration:
        """Minimal fake to prevent ImportError when EvoAgentX imports aliyun_model.
        We won't actually call this in your flow."""
        @staticmethod
        def call(*args, **kwargs):
            class _Resp:
                # mimic a minimal response shape if ever touched
                output = {"text": ""}
            return _Resp()

    _dashscope = types.ModuleType("dashscope")
    _dashscope.Generation = _DummyGeneration
    sys.modules["dashscope"] = _dashscope
# -------------------------------------------------------------------------------


# === HARD STUBS needed by EvoAgentX (must be at the very top) ===
import sys, types

# --- litellm stub (provide completion/acompletion + token utils) ---
try:
    from litellm import completion, acompletion, token_counter, cost_per_token  # if available
except Exception:
    class _Resp:
        def __init__(self, content=""):
            self.choices = [
                type("Choice", (), {
                    "message": type("Msg", (), {"content": content})()
                })()
            ]
    async def acompletion(*args, **kwargs): return _Resp("")
    def completion(*args, **kwargs): return _Resp("")
    def token_counter(text, model=None, **kwargs):
        try: n = len(text or "")
        except Exception: n = 0
        return max(1, n // 4)  # ~4 chars/token
    def cost_per_token(*args, **kwargs): return 0.0

    _m = types.ModuleType("litellm")
    _m.completion = completion
    _m.acompletion = acompletion
    _m.token_counter = token_counter
    _m.cost_per_token = cost_per_token
    sys.modules["litellm"] = _m

# --- overdue stub (timeout context) ---
try:
    import overdue  # real package kung meron
except ModuleNotFoundError:
    class _NoopTimeout:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    sys.modules["overdue"] = type("overdue", (), {"timeout_set_to": _NoopTimeout})()

# --- Tkinter stubs (para hindi maghanap ng X/GUI) ---
if 'tkinter' not in sys.modules:
    sys.modules['tkinter'] = types.ModuleType('tkinter')
    sys.modules['tkinter.ttk'] = types.ModuleType('tkinter.ttk')
    sys.modules['tkinter.filedialog'] = types.ModuleType('tkinter.filedialog')

# --- llama_index minimal stub tree (schema, embeddings, graph_stores, azure_openai) ---
ll_pkg = types.ModuleType("llama_index")
ll_core = types.ModuleType("llama_index.core")
ll_schema = types.ModuleType("llama_index.core.schema")
ll_core_emb = types.ModuleType("llama_index.core.embeddings")
ll_core_graph = types.ModuleType("llama_index.core.graph_stores")
ll_core_graph_types = types.ModuleType("llama_index.core.graph_stores.types")
ll_core_graph_simple = types.ModuleType("llama_index.core.graph_stores.simple")
ll_emb = types.ModuleType("llama_index.embeddings")
ll_az = types.ModuleType("llama_index.embeddings.azure_openai")

class BaseNode:
    def __init__(self, text: str = "", id_: str | None = None, metadata: dict | None = None, **kwargs):
        self.text = text; self.id_ = id_ or "stub"; self.metadata = metadata or {}
class TextNode(BaseNode): pass
class ImageNode(BaseNode):
    def __init__(self, image: bytes | None = None, **kwargs):
        super().__init__(**kwargs); self.image = image
class RelatedNodeInfo:
    def __init__(self, node_id: str | None = None, metadata: dict | None = None, **kwargs):
        self.node_id = node_id or "stub-related"; self.metadata = metadata or {}
class NodeWithScore:
    def __init__(self, node: object, score: float = 0.0, **kwargs):
        self.node = node; self.score = score
ll_schema.NodeWithScore = NodeWithScore
ll_schema.TextNode = TextNode
ll_schema.ImageNode = ImageNode
ll_schema.RelatedNodeInfo = RelatedNodeInfo

class BaseEmbedding:
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]
ll_core_emb.BaseEmbedding = BaseEmbedding

class _StubGraphStore:
    def __init__(self, *args, **kwargs): self._nodes = {}; self._edges = []
    def add_node(self, node_id: str, **kwargs): self._nodes[node_id] = kwargs
    def add_nodes(self, nodes):
        for n in nodes: self.add_node(getattr(n, "id_", "node"), obj=n)
    def add_edge(self, src: str, dst: str, **kwargs): self._edges.append((src, dst, kwargs))
    def get(self, *args, **kwargs): return None
    def get_all(self, *args, **kwargs): return []
    def query(self, *args, **kwargs): return []
    def put(self, *args, **kwargs): pass
ll_core_graph_types.GraphStore = _StubGraphStore
ll_core_graph_simple.GraphStore = _StubGraphStore

class AzureOpenAIEmbedding:
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]
class AzureOpenAIEmbeddingModel:
    text_embedding_3_large = "text-embedding-3-large"
    text_embedding_3_small = "text-embedding-3-small"
ll_az.AzureOpenAIEmbedding = AzureOpenAIEmbedding
ll_az.AzureOpenAIEmbeddingModel = AzureOpenAIEmbeddingModel

sys.modules['llama_index'] = ll_pkg
sys.modules['llama_index.core'] = ll_core
sys.modules['llama_index.core.schema'] = ll_schema
sys.modules['llama_index.core.embeddings'] = ll_core_emb
sys.modules['llama_index.core.graph_stores'] = ll_core_graph
sys.modules['llama_index.core.graph_stores.types'] = ll_core_graph_types
sys.modules['llama_index.core.graph_stores.simple'] = ll_core_graph_simple
sys.modules['llama_index.embeddings'] = ll_emb
sys.modules['llama_index.embeddings.azure_openai'] = ll_az

ll_pkg.core = ll_core
ll_core.schema = ll_schema
ll_core.embeddings = ll_core_emb
ll_core.graph_stores = ll_core_graph
ll_core_graph.types = ll_core_graph_types
ll_core_graph.simple = ll_core_graph_simple
ll_pkg.embeddings = ll_emb
ll_emb.azure_openai = ll_az
# === END HARD STUBS ===


# ===== APP LOGIC =====
import os
from dotenv import load_dotenv

# EvoAgentX (OpenAILLM -> Ollama via OpenAI-compatible endpoint)
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
from evoagentx.actions.code_extraction import CodeExtraction
from evoagentx.actions.code_verification import CodeVerification
from evoagentx.core.module_utils import extract_code_blocks

load_dotenv()

# === Ollama OpenAI-compatible endpoint ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://217.15.175.196:11434/v1").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Force OpenAI SDK to talk to Ollama (dummy key is fine for Ollama)
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

    # === Goal ===
    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_directory = "examples/output/tetris_game"

    # === Build workflow ===
    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    # Optional visual (safe-guarded)
    try:
        workflow_graph.display()
    except Exception as e:
        print(f"[warn] workflow_graph.display() skipped: {e}")

    # === Agents ===
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(
        workflow_graph,
        llm_config=llm.config
    )

    # === Execute workflow ===
    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    # === Verification (reuse same Ollama) ===
    code_verifier = CodeVerification()
    output = code_verifier.execute(
        llm=verification_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

    # === Extraction ===
    os.makedirs(target_directory, exist_ok=True)
    code_blocks = extract_code_blocks(output)
    if len(code_blocks) == 1:
        file_path = os.path.join(target_directory, "index.html")
        with open(file_path, "w", encoding="utf-8") as f:
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
            print("You can open this HTML file in a browser to play the Tetris game")
        else:
            print("This is the main entry point for your application")

if __name__ == "__main__":
    main()
