import os
import yaml
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from ..tools.llm_graph_transformer import LLMGraphTransformer
import logging
from arango import ArangoClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphConstructor:
    """Constructs and manages knowledge graphs from documents."""
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        llm: Optional[BaseLanguageModel] = None
    ):
        """Initialize the graph constructor with configuration."""
        self.config = self._load_config(config_path)
        self.llm = llm or self._initialize_llm()
        self.db_client = None
        self.graph = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _initialize_llm(self) -> BaseLanguageModel:
        """Initialize the OpenRouter LLM for graph construction."""
        try:
            from langchain.llms.base import LLM
            from typing import Optional, List, Mapping, Any
            from pydantic import Field, ConfigDict
            from ..tools.openrouter_client import OpenRouterClient

            class OpenRouterLLM(LLM):
                """Custom LLM that uses OpenRouterClient"""
                model_config = ConfigDict(
                    arbitrary_types_allowed=True,
                    extra='allow',
                    validate_assignment=True,
                    protected_namespaces=()
                )
                
                client: Any = Field(default=None, description="OpenRouter client instance")
                system_prompt: str = Field(default="", description="System prompt for the LLM")
                
                def __init__(self, **kwargs):
                    logger.debug("Initializing OpenRouterLLM with kwargs: %s", kwargs)
                    super().__init__(**kwargs)
                    self.client = OpenRouterClient(component_type='extract_model')
                    logger.debug("OpenRouterLLM initialized successfully")
                
                def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                    """Call the OpenRouter API and return the response."""
                    response = self.client.get_completion(
                        prompt=prompt,
                        system_prompt=self.system_prompt or """You are a specialized entity extractor for sociopolitical and historical analysis.
                        Extract named entities, dates, events, and their relationships with high accuracy.""",
                        temperature=0.0
                    )
                    if response.get("error"):
                        raise ValueError(f"OpenRouter API error: {response['error']}")
                    return response["content"]
                
                @property
                def _llm_type(self) -> str:
                    return "openrouter"
                
                @property
                def _identifying_params(self) -> Mapping[str, Any]:
                    return {"model": self.client.model}
            
            # Initialize our custom LLM
            llm = OpenRouterLLM()
            
            # Update system prompt for sociopolitical context
            llm.system_prompt = """
            You are a specialized entity extractor for sociopolitical and historical analysis.
            Your task is to identify and extract:
            1. Named Entities: People, Organizations, Locations, Events
            2. Temporal Information: Dates, Time Periods, Historical Eras
            3. Relationships: Political, Social, Economic connections between entities
            4. Key Concepts: Ideologies, Movements, Policies
            
            Extract as much detail as possible while maintaining accuracy. Pay special attention to:
            - Historical figures and their roles
            - Organizations and institutions
            - Significant events and their dates
            - Causal relationships between events
            - Power dynamics and social structures
            
            Do not restrict yourself to predefined categories. The goal is to capture the rich
            interconnections in sociopolitical and historical contexts.
            """
            
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing OpenRouter LLM: {str(e)}")
            raise

    def _initialize_db_client(self) -> ArangoClient:
        """Initialize the ArangoDB client."""
        try:
            db_config = self.config['arangodb']
            client = ArangoClient(
                hosts=f"http://{db_config['host']}:{db_config['port']}"
            )
            
            # Connect to system database first
            sys_db = client.db(
                '_system',
                username=db_config['username'],
                password=db_config['password']
            )
            
            # Create application database if it doesn't exist
            if not sys_db.has_database(db_config['database']):
                sys_db.create_database(db_config['database'])
                logger.info(f"Created database: {db_config['database']}")
            
            return client
            
        except Exception as e:
            logger.error(f"Error initializing database client: {str(e)}")
            raise

    def _initialize_graph(self) -> Neo4jGraph:
        """Initialize the graph database."""
        try:
            db_config = self.config['arangodb']
            db = self.db_client.db(
                db_config['database'],
                username=db_config['username'],
                password=db_config['password']
            )
            
            # Create or get the graph
            if not db.has_graph('knowledge_graph'):
                graph = db.create_graph('knowledge_graph')
                logger.info("Created new knowledge graph")
            else:
                graph = db.graph('knowledge_graph')
                logger.info("Connected to existing knowledge graph")
            
            # Ensure vertex collections exist
            if not graph.has_vertex_collection('nodes'):
                graph.create_vertex_collection('nodes')
            
            # Ensure edge collections exist
            if not graph.has_edge_definition('relationships'):
                graph.create_edge_definition(
                    edge_collection='relationships',
                    from_vertex_collections=['nodes'],
                    to_vertex_collections=['nodes']
                )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error initializing graph: {str(e)}")
            raise

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        try:
            # Delete all relationships first
            self.graph.edge_collection('relationships').truncate()
            logger.info("Cleared all relationships")
            
            # Then delete all nodes
            self.graph.vertex_collection('nodes').truncate()
            logger.info("Cleared all nodes")
            
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            raise

    def insert_node(self, node: Dict[str, Any]) -> None:
        """Insert a node into the graph if it doesn't already exist."""
        try:
            node_collection = self.graph.vertex_collection('nodes')
            
            if not node_collection.has(node['id']):
                node_collection.insert({
                    '_key': node['id'],
                    'type': node['type'],
                    **(node.get('properties') or {})
                })
                logger.debug(f"Inserted node: {node['id']}")
                
        except Exception as e:
            logger.error(f"Error inserting node: {str(e)}")
            raise

    def insert_relationship(self, relationship: Dict[str, Any]) -> None:
        """Insert a relationship into the graph if it doesn't already exist."""
        try:
            edge_collection = self.graph.edge_collection('relationships')
            
            source_id = relationship['source']['id']
            target_id = relationship['target']['id']
            edge_key = f"{source_id}_{relationship['type']}_{target_id}"
            
            if not edge_collection.has(edge_key):
                edge_collection.insert({
                    '_key': edge_key,
                    '_from': f"nodes/{source_id}",
                    '_to': f"nodes/{target_id}",
                    'type': relationship['type'],
                    **(relationship.get('properties') or {})
                })
                logger.debug(f"Inserted relationship: {edge_key}")
                
        except Exception as e:
            logger.error(f"Error inserting relationship: {str(e)}")
            raise

    def construct_graph(
        self,
        documents: List[Document],
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> None:
        """
        Construct a knowledge graph from documents using the LLM transformer.
        
        Args:
            documents: List of input documents
            allowed_nodes: List of allowed node types
            allowed_relationships: List of allowed relationship types
            batch_size: Size of document batches for processing
        """
        try:
            # Initialize the LLM transformer without restrictions for sociopolitical context
            transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=[],  # Empty list to allow any node type
                allowed_relationships=[],  # Empty list to allow any relationship type
                strict_mode=False,  # Don't restrict entity/relationship types
                ignore_tool_usage=True,  # Use simpler extraction without function calling
                node_properties=False,  # Don't extract node properties
                relationship_properties=False  # Don't extract relationship properties
            )
            
            # Process documents in batches and collect all graph documents
            all_graph_documents = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                graph_documents = transformer.convert_to_graph_documents(batch)
                all_graph_documents.extend(graph_documents)
                logger.info(f"Processed batch {i//batch_size + 1}")
            
            logger.info(f"Extracted entities and relationships from {len(documents)} documents")
            
            # Initialize database connection only when we have data to store
            if any(doc.nodes or doc.relationships for doc in all_graph_documents):
                try:
                    if self.db_client is None:
                        self.db_client = self._initialize_db_client()
                    if self.graph is None:
                        self.graph = self._initialize_graph()
                        
                    # Insert all nodes and relationships
                    for graph_doc in all_graph_documents:
                        # Insert nodes
                        if graph_doc.nodes:
                            for node in graph_doc.nodes:
                                self.insert_node({
                                    'id': node.id,
                                    'type': node.type,
                                    'properties': node.properties
                                })
                        
                        # Insert relationships
                        if graph_doc.relationships:
                            for rel in graph_doc.relationships:
                                self.insert_relationship({
                                    'source': {'id': rel.source.id},
                                    'target': {'id': rel.target.id},
                                    'type': rel.type,
                                    'properties': rel.properties
                                })
                    
                    logger.info("Successfully stored graph in database")
                except Exception as e:
                    logger.warning(f"Failed to store graph in database: {str(e)}")
                    logger.info("Continuing with in-memory graph only")
            
            logger.info("Completed graph construction")
            
        except Exception as e:
            logger.error(f"Error constructing graph: {str(e)}")
            raise

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query on the graph database.
        
        Args:
            query: AQL query string
            
        Returns:
            List of query results
        """
        try:
            db = self.db_client.db(
                self.config['arangodb']['database'],
                username=self.config['arangodb']['username'],
                password=self.config['arangodb']['password']
            )
            
            cursor = db.aql.execute(query)
            results = [doc for doc in cursor]
            
            logger.info(f"Query executed successfully: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise