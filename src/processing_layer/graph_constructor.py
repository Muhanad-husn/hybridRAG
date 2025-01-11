import os
import yaml
import json
import networkx as nx
import pandas as pd
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from ..tools.llm_graph_transformer import LLMGraphTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphConstructor:
    """Constructs and manages knowledge graphs from documents using NetworkX."""
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        llm: Optional[BaseLanguageModel] = None
    ):
        """Initialize the graph constructor with configuration."""
        self.config = self._load_config(config_path)
        self.llm = llm or self._initialize_llm()
        self.graph = nx.DiGraph()  # Using directed graph for relationships
        self._setup_storage_paths()
        self._load_existing_graph()
        
    def _setup_storage_paths(self) -> None:
        """Setup paths for CSV storage."""
        try:
            # Ensure graphs directory exists
            graphs_dir = os.path.join('data', 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            
            # Set file paths
            self.nodes_file = os.path.join(graphs_dir, 'nodes.csv')
            self.edges_file = os.path.join(graphs_dir, 'edges.csv')
            
            logger.info(f"Storage paths setup: {graphs_dir}")
        except Exception as e:
            logger.error(f"Error setting up storage paths: {str(e)}")
            raise

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

    def _load_existing_graph(self) -> None:
        """Load existing graph from CSV files if they exist."""
        try:
            if os.path.exists(self.nodes_file) and os.path.exists(self.edges_file):
                # Load nodes
                nodes_df = pd.read_csv(self.nodes_file)
                for _, row in nodes_df.iterrows():
                    properties = json.loads(row['properties']) if row['properties'] else {}
                    self.graph.add_node(
                        row['id'],
                        type=row['type'],
                        **properties
                    )

                # Load edges
                edges_df = pd.read_csv(self.edges_file)
                for _, row in edges_df.iterrows():
                    properties = json.loads(row['properties']) if row['properties'] else {}
                    self.graph.add_edge(
                        row['source_id'],
                        row['target_id'],
                        type=row['type'],
                        **properties
                    )
                
                logger.info(f"Loaded existing graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error loading existing graph: {str(e)}")
            raise

    def _save_graph(self) -> None:
        """Save current graph to CSV files."""
        try:
            # Save nodes
            nodes_data = []
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.pop('type', 'unknown')
                nodes_data.append({
                    'id': node_id,
                    'type': node_type,
                    'properties': json.dumps(node_data)
                })
            pd.DataFrame(nodes_data).to_csv(self.nodes_file, index=False)

            # Save edges
            edges_data = []
            for source, target, edge_data in self.graph.edges(data=True):
                edge_type = edge_data.pop('type', 'unknown')
                edges_data.append({
                    'source_id': source,
                    'target_id': target,
                    'type': edge_type,
                    'properties': json.dumps(edge_data)
                })
            pd.DataFrame(edges_data).to_csv(self.edges_file, index=False)

            logger.info("Graph saved to CSV files successfully")
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            raise

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        try:
            self.graph.clear()
            # Clear CSV files if they exist
            if os.path.exists(self.nodes_file):
                os.remove(self.nodes_file)
            if os.path.exists(self.edges_file):
                os.remove(self.edges_file)
            logger.info("Cleared graph and removed CSV files")
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            raise

    def insert_node(self, node: Dict[str, Any]) -> None:
        """Insert a node into the graph if it doesn't already exist."""
        try:
            if not self.graph.has_node(node['id']):
                properties = node.get('properties', {})
                self.graph.add_node(
                    node['id'],
                    type=node['type'],
                    **properties
                )
                self._save_graph()  # Save after each insertion
                logger.debug(f"Inserted node: {node['id']}")
        except Exception as e:
            logger.error(f"Error inserting node: {str(e)}")
            raise

    def insert_relationship(self, relationship: Dict[str, Any]) -> None:
        """Insert a relationship into the graph if it doesn't already exist."""
        try:
            source_id = relationship['source']['id']
            target_id = relationship['target']['id']
            
            if not self.graph.has_edge(source_id, target_id):
                properties = relationship.get('properties', {})
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=relationship['type'],
                    **properties
                )
                self._save_graph()  # Save after each insertion
                logger.debug(f"Inserted relationship: {source_id} -> {target_id}")
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
            
            logger.info("Completed graph construction")
            
        except Exception as e:
            logger.error(f"Error constructing graph: {str(e)}")
            raise

    def query_graph(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the graph using NetworkX functions.
        
        Args:
            query: Dictionary containing query parameters:
                  - type: str (query type: 'neighbors', 'shortest_path', 'subgraph', etc.)
                  - params: Dict (query-specific parameters)
                  
        Returns:
            List of query results
        """
        try:
            query_type = query.get('type', '')
            params = query.get('params', {})
            
            if query_type == 'neighbors':
                node_id = params.get('node_id')
                results = list(self.graph.neighbors(node_id))
            elif query_type == 'shortest_path':
                source = params.get('source')
                target = params.get('target')
                results = nx.shortest_path(self.graph, source, target)
            elif query_type == 'subgraph':
                nodes = params.get('nodes', [])
                subgraph = self.graph.subgraph(nodes)
                results = [{'nodes': list(subgraph.nodes(data=True)), 
                          'edges': list(subgraph.edges(data=True))}]
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            logger.info(f"Query executed successfully: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise