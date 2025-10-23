# ---- STUBS for LlamaIndex bits EvoAgentX imports ----
# We create fake modules so EvoAgentX can import them without the real llama-index.
import sys, types

ll_pkg = types.ModuleType("llama_index"); ll_pkg.__path__ = []  # mark as pkg

# subpackages
ll_core = types.ModuleType("llama_index.core"); ll_core.__path__ = []
ll_schema = types.ModuleType("llama_index.core.schema")
ll_core_emb = types.ModuleType("llama_index.core.embeddings")
ll_core_graph = types.ModuleType("llama_index.core.graph_stores"); ll_core_graph.__path__ = []
ll_core_graph_types = types.ModuleType("llama_index.core.graph_stores.types")
ll_core_graph_simple = types.ModuleType("llama_index.core.graph_stores.simple")

ll_emb = types.ModuleType("llama_index.embeddings"); ll_emb.__path__ = []
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
    def __init__(self, *args, **kwargs): pass
    def get_text_embedding(self, text: str): return [0.0]
    def get_text_embedding_batch(self, texts): return [[0.0] for _ in texts]

ll_core_emb.BaseEmbedding = BaseEmbedding

# --- core.graph_stores GraphStore stub (shared) ---
class _StubGraphStore:
    def __init__(self, *args, **kwargs):
        self._nodes = {}
        self._edges = []
    def add_node(self, node_id: str, **kwargs):
        self._nodes[node_id] = kwargs
    def add_nodes(self, nodes):
        for n in nodes:
            self.add_node(getattr(n, "id_", "node"), obj=n)
    def add_edge(self, src: str, dst: str, **kwargs):
        self._edges.append((src, dst, kwargs))
    def get(self, *args, **kwargs):
        return None
    def query(self, *args, **kwargs):
        return []

# types.GraphStore
ll_core_graph_types.GraphStore = _StubGraphStore
# simple.GraphStore
ll_core_graph_simple.GraphStore = _StubGraphStore

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
sys.modules['llama_index.core.graph_stores'] = ll_core_graph
sys.modules['llama_index.core.graph_stores.types'] = ll_core_graph_types
sys.modules['llama_index.core.graph_stores.simple'] = ll_core_graph_simple
sys.modules['llama_index.embeddings'] = ll_emb
sys.modules['llama_index.embeddings.azure_openai'] = ll_az

# Link hierarchy (attribute access)
ll_pkg.core = ll_core
ll_core.schema = ll_schema
ll_core.embeddings = ll_core_emb
ll_core.graph_stores = ll_core_graph
ll_core_graph.types = ll_core_graph_types
ll_core_graph.simple = ll_core_graph_simple

ll_pkg.embeddings = ll_emb
ll_emb.azure_openai = ll_az
# ------------------------------------------------------
