# =========================
#  HARD STUBS (TOP-OF-FILE)
# =========================
import sys, types

# --- HARD STUB: dashscope (Aliyun) ---
try:
    from dashscope import Generation
except Exception:
    class _DummyGeneration:
        @staticmethod
        def call(*args, **kwargs):
            class _Resp: output = {"text": ""}
            return _Resp()
    _dashscope = types.ModuleType("dashscope")
    _dashscope.Generation = _DummyGeneration
    sys.modules["dashscope"] = _dashscope
# -------------------------------------------------------------------------------

# --- HARD STUB: Neo4jGraphStoreWrapper (avoid real neo4j) ---
_neo4j_mod = types.ModuleType("evoagentx.storages.graph_stores.neo4j")
class Neo4jGraphStoreWrapper:
    def __init__(self, *args, **kwargs): pass
    def as_graph_store(self):
        from llama_index.core.graph_stores.simple import GraphStore
        return GraphStore()
_neo4j_mod.Neo4jGraphStoreWrapper = Neo4jGraphStoreWrapper
sys.modules["evoagentx.storages.graph_stores.neo4j"] = _neo4j_mod
# -------------------------------------------------------------------------------

# --- litellm stub (completion/acompletion + token utils) ---
try:
    from litellm import completion, acompletion, token_counter, cost_per_token
except Exception:
    class _Resp:
        def __init__(self, content=""):
            self.choices = [type("Choice", (), {
                "message": type("Msg", (), {"content": content})()
            })()]
    async def acompletion(*args, **kwargs): return _Resp("")
    def completion(*args, **kwargs): return _Resp("")
    def token_counter(text, model=None, **kwargs):
        try: n = len(text or "")
        except Exception: n = 0
        return max(1, n // 4)
    def cost_per_token(*args, **kwargs): return 0.0
    _m = types.ModuleType("litellm")
    _m.completion = completion
    _m.acompletion = acompletion
    _m.token_counter = token_counter
    _m.cost_per_token = cost_per_token
    sys.modules["litellm"] = _m
# -------------------------------------------------------------------------------

# --- overdue stub (timeout context) ---
try:
    import overdue
except ModuleNotFoundError:
    class _NoopTimeout:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
    sys.modules["overdue"] = type("overdue", (), {"timeout_set_to": _NoopTimeout})()
# -------------------------------------------------------------------------------

# --- Tkinter headless stubs ---
if 'tkinter' not in sys.modules:
    sys.modules['tkinter'] = types.ModuleType('tkinter')
    sys.modules['tkinter.ttk'] = types.ModuleType('tkinter.ttk')
    sys.modules['tkinter.filedialog'] = types.ModuleType('tkinter.filedialog')
# -------------------------------------------------------------------------------

# --- llama_index mega-stub tree (schema, embeddings, graph_stores, vector_stores, storage, indices, azure_openai) ---
ll_pkg = types.ModuleType("llama_index")
ll_core = types.ModuleType("llama_index.core")
ll_schema = types.ModuleType("llama_index.core.schema")
ll_core_emb = types.ModuleType("llama_index.core.embeddings")
ll_core_graph = types.ModuleType("llama_index.core.graph_stores")
ll_core_graph_types = types.ModuleType("llama_index.core.graph_stores.types")
ll_core_graph_simple = types.ModuleType("llama_index.core.graph_stores.simple")
ll_core_vector = types.ModuleType("llama_index.core.vector_stores")
ll_core_vector_types = types.ModuleType("llama_index.core.vector_stores.types")
ll_core_storage = types.ModuleType("llama_index.core.storage")
ll_core_indices = types.ModuleType("llama_index.core.indices")
ll_core_indices_base = types.ModuleType("llama_index.core.indices.base")

ll_emb = types.ModuleType("llama_index.embeddings")
ll_az = types.ModuleType("llama_index.embeddings.azure_openai")

# Non-core vector stores path (faiss)
ll_vector_pkg = types.ModuleType("llama_index.vector_stores")
ll_vector_faiss = types.ModuleType("llama_index.vector_stores.faiss")

# ----- core.schema -----
class BaseNode:
    def __init__(self, text: str = "", id_: str | None = None, metadata: dict | None = None, **kwargs):
        self.text = text
        self.id_ = id_ or "stub"
        self.metadata = metadata or {}
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

# NEW: QueryBundle + extras that evoagentx may touch
class QueryBundle:
    def __init__(self, query_str: str = "", embedding=None, custom_embedding_strs=None, **kwargs):
        self.query_str = query_str
        self.embedding = embedding
        self.custom_embedding_strs = custom_embedding_strs or []
class Document(BaseNode): pass
class MetadataMode:
    ALL = "ALL"
    NONE = "NONE"

ll_schema.BaseNode = BaseNode
ll_schema.TextNode = TextNode
ll_schema.ImageNode = ImageNode
ll_schema.RelatedNodeInfo = RelatedNodeInfo
ll_schema.NodeWithScore = NodeWithScore
ll_schema.QueryBundle = QueryBundle
ll_schema.Document = Document
ll_schema.MetadataMode = MetadataMode

# ----- core.embeddings -----
class BaseEmbedding:
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]
ll_core_emb.BaseEmbedding = BaseEmbedding

# ----- core.graph_stores (types + simple) -----
class _StubGraphStore:
    def __init__(self, *args, **kwargs):
        self._nodes = {}; self._edges = []
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

# >>>>>> FIX: add Relation, EntityNode, ChunkNode <<<<<<
class Relation:
    def __init__(self, source_id: str = "", target_id: str = "", type: str = "", metadata: dict | None = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.metadata = metadata or {}

class EntityNode:
    def __init__(self, id_: str = "", label: str = "", properties: dict | None = None, metadata: dict | None = None):
        self.id_ = id_
        self.label = label
        self.properties = properties or {}
        self.metadata = metadata or {}

class ChunkNode:
    def __init__(self, id_: str = "", text: str = "", metadata: dict | None = None):
        self.id_ = id_
        self.text = text
        self.metadata = metadata or {}
# expose
ll_core_graph_types.Relation = Relation
ll_core_graph_types.EntityNode = EntityNode
ll_core_graph_types.ChunkNode = ChunkNode
# -------------------------------------------------------------------

# ----- core.vector_stores (types) -----
class BasePydanticVectorStore: pass
class VectorStore: pass
ll_core_vector_types.BasePydanticVectorStore = BasePydanticVectorStore
ll_core_vector_types.VectorStore = VectorStore

# ----- non-core vector_stores.faiss -----
class FaissMapVectorStore:
    def __init__(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def query(self, *args, **kwargs): return []
ll_vector_faiss.FaissMapVectorStore = FaissMapVectorStore

# ----- core.storage -----
class StorageContext:
    def __init__(self, *args, **kwargs): pass
ll_core_storage.StorageContext = StorageContext

# ----- core.indices.base -----
class BaseIndex:
    def __init__(self, *args, **kwargs): pass
    def as_retriever(self, *args, **kwargs):
        class _R:
            def retrieve(self, *a, **k): return []
        return _R()
ll_core_indices_base.BaseIndex = BaseIndex
ll_core_indices.base = ll_core_indices_base

# ----- VectorStoreIndex (exposed at llama_index.core) -----
class VectorStoreIndex:
    def __init__(self, *args, **kwargs): pass
    @classmethod
    def from_documents(cls, *args, **kwargs):
        return cls()
    def as_retriever(self, *args, **kwargs):
        class _R:
            def retrieve(self, *a, **k): return []
        return _R()

# ----- embeddings.azure_openai -----
class AzureOpenAIEmbedding:
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]
class AzureOpenAIEmbeddingModel:
    text_embedding_3_large = "text-embedding-3-large"
    text_embedding_3_small = "text-embedding-3-small"
ll_az.AzureOpenAIEmbedding = AzureOpenAIEmbedding
ll_az.AzureOpenAIEmbeddingModel = AzureOpenAIEmbeddingModel

# ----- register modules -----
sys.modules['llama_index'] = ll_pkg
sys.modules['llama_index.core'] = ll_core
sys.modules['llama_index.core.schema'] = ll_schema
sys.modules['llama_index.core.embeddings'] = ll_core_emb
sys.modules['llama_index.core.graph_stores'] = ll_core_graph
sys.modules['llama_index.core.graph_stores.types'] = ll_core_graph_types
sys.modules['llama_index.core.graph_stores.simple'] = ll_core_graph_simple
sys.modules['llama_index.core.vector_stores'] = ll_core_vector
sys.modules['llama_index.core.vector_stores.types'] = ll_core_vector_types
sys.modules['llama_index.core.storage'] = ll_core_storage
sys.modules['llama_index.core.indices'] = ll_core_indices
sys.modules['llama_index.core.indices.base'] = ll_core_indices_base

sys.modules['llama_index.embeddings'] = ll_emb
sys.modules['llama_index.embeddings.azure_openai'] = ll_az

sys.modules['llama_index.vector_stores'] = ll_vector_pkg
sys.modules['llama_index.vector_stores.faiss'] = ll_vector_faiss

# ----- link attribute hierarchy -----
ll_pkg.core = ll_core
ll_core.schema = ll_schema
ll_core.embeddings = ll_core_emb
ll_core.graph_stores = ll_core_graph
ll_core.vector_stores = ll_core_vector
ll_core.storage = ll_core_storage
ll_core.indices = ll_core_indices
ll_core_graph.types = ll_core_graph_types
ll_core_graph.simple = ll_core_graph_simple
ll_core_vector.types = ll_core_vector_types
ll_core.VectorStoreIndex = VectorStoreIndex
ll_core.Document = Document
ll_core.QueryBundle = QueryBundle

ll_pkg.embeddings = ll_emb
ll_emb.azure_openai = ll_az

ll_pkg.vector_stores = ll_vector_pkg
ll_vector_pkg.faiss = ll_vector_faiss
# =========================
#  END HARD STUBS
# =========================


# =============
#  APP LOGIC
# =============
import os
from dotenv import load_dotenv

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
    llm = build_llm(max_tokens=16000, temperature=0.2)
    verification_llm = build_llm(max_tokens=20000, temperature=0.0)

    goal = "Generate html code for the Tetris game that can be played in the browser."
    target_directory = "examples/output/tetris_game"

    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    try:
        workflow_graph.display()
    except Exception as e:
        print(f"[warn] workflow_graph.display() skipped: {e}")

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm.config)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute()

    code_verifier = CodeVerification()
    output = code_verifier.execute(
        llm=verification_llm,
        inputs={"requirements": goal, "code": output}
    ).verified_code

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
        inputs={"code_string": output, "target_directory": target_directory}
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
