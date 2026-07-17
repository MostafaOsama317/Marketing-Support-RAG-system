import os
from dotenv import load_dotenv

from wasabi import msg
from weaviate import Client

from Verba.goldenverba.components.interfaces import VerbaComponent
from backend.components.document import Document
from backend.server.types import FileConfig
from backend.components.types import InputConfig

load_dotenv()

class RagComponent:
    """
    Base Class for Rag Readers, Chunkers, Embedders, Retrievers, and Generators.
    """

    def __init__(self):
        self.name = ""
        self.requires_env = [] # List of required environment variables for the component
        self.requires_library = [] # List of required libraries for the component
        self.description = "" # Description of the component
        self.config = {} # Settings for the component, can be a dictionary of InputConfig or FileConfig
        self.type = "" # Type of component: reader, chunker, embedder, retriever, generator
    
    def get_meta(self, envs, libs) -> dict: 
        """
        Get metadata about the component, including its name,
        required environment variables and libraries, 
        description, type, and configuration and put it into a dictionary for frontend.
        """

        if len(self.config) > 0:
            config = {_c: self.config[_c].model_dump() for _c in self.config}
        else:
            config = {}

        return {
            "name": self.name,
            "variables": self.requires_env,
            "library": self.requires_library,
            "description": self.description,
            "type": self.type,
            "config": config,
            "available": self.check_available(envs, libs),
        }
    
    def check_available(self, envs, libs) -> bool:
        """
        Check if the component is available based on 
        the required environment variables and libraries.
        """
        if self.requires_env:
            for _env in self.requires_env:
                if _env not in envs or not envs.get(_env, False):
                    return False
        if self.requires_library:
            for _lib in self.requires_library:
                if _lib not in libs or not libs.get(_lib, False):
                    return False
        return True

class Reader(RagComponent):
    """
    Interface for Rag Readers.
    """
    def __init__(self):
        super().__init__()
        self.type = "FILE"  # "URL"
        self.extension = ["txt", "md", "mdx", "py", "ts", "tsx", "js", "go", "css"]

    async def load(self, config: dict, fileConfig: FileConfig) -> list[Document]:
        """Convert fileConfig into Verba Documents
        @parameter: fileConfig: FileConfig - FileConfiguration sent by the frontend
        @returns list[Document] - Verba documents
        """
        raise NotImplementedError("load method must be implemented by a subclass.")
    
class Embedding(RagComponent):
    """
    Interface for Rag Embedder Components.
    """
    def __init__(self):
        super().__init__()
        self.max_batch_size = 128

    async def vectorize(self, config: dict, content: list[str]) -> list[float]:
        """Embed verba documents and its chunks to Weaviate
        @parameter: config : dict - Embedder Configuration
        @parameter: content : list[str] - List of strings to embed
        @return: list[float] - List of embeddings
        """
        raise NotImplementedError("embed method must be implemented by a subclass.")
    

class chunker(RagComponent):
    """
    Interface for Rag Chunker Components.
    """
    def __init__(self):
        super().__init__()
        self.config = {}

    async def chunk(
        self,
        config: dict,
        documents: list[Document],
        embedder: Embedding | None = None,
        embedder_config: dict | None = None,
    ) -> list[Document]:
        """Split Verba documents into chunks.
        @parameter: config : dict - Chunker Configuration
        @parameter: documents : list[Document] - List of Verba documents to chunk
        @parameter: embedder : Embedding | None - (Optional) Selected Embedder if the Chunker requires vectorization
        @parameter: embedder_config : dict | None - (Optional) Embedder Configuration
        @return: list[Documents]
        """
        raise NotImplementedError("chunk method must be implemented by a subclass.")
    

class Retriever(RagComponent):
    """
    Interface for Verba Retrievers.
    """

    def __init__(self):
        super().__init__()
        self.config["Suggestion"] = InputConfig(
            type="bool",
            value=True,
            description="Enable Autocomplete Suggestions",
            values=[],
        )

    async def retrieve(
        self,
        client,
        query,
        vector,
        config,
        weaviate_manager,
        embedder,
        labels,
        document_uuids,
    ):

        raise NotImplementedError("retrieve method must be implemented by a subclass.")


class Generator(VerbaComponent):
    """
    Interface for Verba Generators.
    """

    def __init__(self):
        super().__init__()
        self.context_window = 5000
        default_prompt = "You are marketing assistant, a chatbot for Retrieval Augmented Generation (RAG). You will receive a user query and context pieces that have a semantic similarity to that query. Please answer these user queries only with the provided context. Mention documents you used from the context if you use them to reduce hallucination. If the provided documentation does not provide enough information, say so. If the user asks questions about you as a chatbot specifially, answer them naturally. If the answer requires code examples encapsulate them with ```programming-language-name ```. Don't do pseudo-code."
        prompt = os.getenv("SYSYEM_MESSAGE_PROMPT", default_prompt)
        # ui clickable prompt to edit system message
        self.config["System Message"] = InputConfig( 
            type="textarea",
            value=prompt,
            description="System Message",
            values=[],
        )


    async def generate_stream(
        self,
        queries: list[str],
        context: list[str],
        conversation: dict = None,
    ):
        """Generate a stream of response dicts based on a list of queries and list of contexts, and includes conversational context
        @parameter: queries : list[str] - List of queries
        @parameter: context : list[str] - List of contexts
        @parameter: conversation : dict - Conversational context
        @returns Iterator[dict] - Token response generated by the Generator in this format {system:TOKEN, finish_reason:stop or empty}.
        """
        if conversation is None:
            conversation = {}
        raise NotImplementedError(
            "generate_stream method must be implemented by a subclass."
        )

    def prepare_messages(
        self, queries: list[str], context: list[str], conversation: dict[str, str]
    ) -> any:
        """
        Prepares a list of messages formatted for a Retrieval Augmented Generation chatbot system, including system instructions, previous conversation, and a new user query with context.

        @parameter queries: A list of strings representing the user queries to be answered.
        @parameter context: A list of strings representing the context information provided for the queries.
        @parameter conversation: A list of previous conversation messages that include the role and content.

        @returns A list or of message dictionaries or whole prompts formatted for the chatbot. This includes an initial system message, the previous conversation messages, and the new user query encapsulated with the provided context.

        Each message in the list is a dictionary with 'role' and 'content' keys, where 'role' is either 'system' or 'user', and 'content' contains the relevant text. This will depend on the LLM used.
        """
        raise NotImplementedError(
            "prepare_messages method must be implemented by a subclass."
        )