import os
import yaml
import json
import networkx as nx
import pandas as pd
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
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
        self._setup_storage_paths()
        self.graph = nx.DiGraph()  # Using directed graph for relationships
        if os.path.exists(self.nodes_file) and os.path.exists(self.edges_file):
            self._load_existing_graph()
        logger.info(f"Graph initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _setup_storage_paths(self) -> None:
        """Setup paths for CSV storage and initialize files."""
        try:
            # Ensure graphs directory exists
            graphs_dir = os.path.join('data', 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            
            # Set file paths
            self.nodes_file = os.path.join(graphs_dir, 'nodes.csv')
            self.edges_file = os.path.join(graphs_dir, 'edges.csv')
            
            # Initialize CSV files with headers if they don't exist
            if not os.path.exists(self.nodes_file):
                pd.DataFrame(columns=['id', 'type', 'properties']).to_csv(self.nodes_file, index=False)
                logger.info("Initialized nodes.csv with headers")
                
            if not os.path.exists(self.edges_file):
                pd.DataFrame(columns=['source_id', 'target_id', 'type', 'properties']).to_csv(self.edges_file, index=False)
                logger.info("Initialized edges.csv with headers")
            
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
                        system_prompt=self.system_prompt,
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
            1. Named Entities:
               - People (individuals, historical figures)
               - Organizations (institutions, companies, governments)
               - Locations (countries, cities, regions)
               - Events (historical events, incidents, developments)
               - Groups (ethnic groups, sects, factions, movements)
            2. Temporal Information:
               - Dates (specific dates, years)
               - Time Periods (eras, decades, phases)
            3. Relationships:
               - Political (LEADS, GOVERNS, REPRESENTS)
               - Social (MEMBER_OF, ALLIED_WITH, OPPOSED)
               - Economic (CONTROLS, SUPPORTS, FUNDS)
               - Military (FIGHTS_AGAINST, SUPPORTS_MILITARY, INTERVENES)
            4. Key Concepts:
               - Ideologies
               - Policies
               - Movements
               - Social phenomena

            Extract as much detail as possible while maintaining accuracy. Pay special attention to:
            - Historical figures and their roles
            - Organizations and institutions
            - Significant events and their dates
            - Causal relationships between events
            - Power dynamics and social structures
            - Group affiliations and factional relationships
            - Military and political alliances
            - Economic and social impacts

            Format your response as a valid JSON object with nodes and relationships arrays.
            Each node must have an id and type.
            Each relationship must have a source, target, and type.
            Use consistent and standardized relationship types in UPPER_CASE.
            
            Remember:
            1. Extract only explicitly stated information - no assumptions or inferences
            2. Preserve exact multi-word entities as they appear in the text
            3. Ensure relationships are clearly supported by the text
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

    def _append_node_to_csv(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Append a single node to the nodes CSV file."""
        try:
            # Create a copy of node_data without modifying the original
            node_type = node_data.get('type', 'unknown')
            properties = {k: v for k, v in node_data.items() if k != 'type'}
            
            node_row = {
                'id': node_id,
                'type': node_type,
                'properties': json.dumps(properties)
            }
            
            # Create file with headers if it doesn't exist
            if not os.path.exists(self.nodes_file):
                pd.DataFrame([node_row]).to_csv(self.nodes_file, index=False)
            else:
                # Append without headers
                pd.DataFrame([node_row]).to_csv(self.nodes_file, mode='a', header=False, index=False)
            
            logger.debug(f"Node {node_id} appended to CSV")
        except Exception as e:
            logger.error(f"Error appending node to CSV: {str(e)}")
            raise

    def _append_edge_to_csv(self, source: str, target: str, edge_data: Dict[str, Any]) -> None:
        """Append a single edge to the edges CSV file."""
        try:
            # Create a copy of edge_data without modifying the original
            edge_type = edge_data.get('type', 'unknown')
            properties = {k: v for k, v in edge_data.items() if k != 'type'}
            
            edge_row = {
                'source_id': source,
                'target_id': target,
                'type': edge_type,
                'properties': json.dumps(properties)
            }
            
            # Create file with headers if it doesn't exist
            if not os.path.exists(self.edges_file):
                pd.DataFrame([edge_row]).to_csv(self.edges_file, index=False)
            else:
                # Append without headers
                pd.DataFrame([edge_row]).to_csv(self.edges_file, mode='a', header=False, index=False)
            
            logger.debug(f"Edge {source}->{target} appended to CSV")
        except Exception as e:
            logger.error(f"Error appending edge to CSV: {str(e)}")
            raise

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph."""
        try:
            # Only clear the graph object, keep CSV files
            self.graph.clear()
            logger.info("Cleared graph (CSV files preserved)")
        except Exception as e:
            logger.error(f"Error clearing graph: {str(e)}")
            raise

    def insert_node(self, node: Dict[str, Any]) -> None:
        """Insert a node into the graph if it doesn't already exist and append to CSV."""
        try:
            if not self.graph.has_node(node['id']):
                properties = node.get('properties', {})
                # Add to NetworkX graph
                self.graph.add_node(
                    node['id'],
                    type=node['type'],
                    **properties
                )
                # Append to CSV immediately
                self._append_node_to_csv(
                    node['id'],
                    {'type': node['type'], **properties}
                )
                logger.debug(f"Inserted node: {node['id']}")
        except Exception as e:
            logger.error(f"Error inserting node: {str(e)}")
            raise

    def insert_relationship(self, relationship: Dict[str, Any]) -> None:
        """Insert a relationship into the graph if it doesn't already exist and append to CSV."""
        try:
            source_id = relationship['source']['id']
            target_id = relationship['target']['id']
            
            if not self.graph.has_edge(source_id, target_id):
                properties = relationship.get('properties', {})
                # Add to NetworkX graph
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=relationship['type'],
                    **properties
                )
                # Append to CSV immediately
                self._append_edge_to_csv(
                    source_id,
                    target_id,
                    {'type': relationship['type'], **properties}
                )
                logger.debug(f"Inserted relationship: {source_id} -> {target_id}")
        except Exception as e:
            logger.error(f"Error inserting relationship: {str(e)}")
            raise

    async def aconstruct_graph(self, documents: List[Document], batch_size: int = 10) -> None:
        """
        Asynchronously construct a knowledge graph from documents using the LLM transformer.
        
        Args:
            documents: List of input documents
            batch_size: Size of document batches for processing
        """
        try:
            # Initialize the LLM transformer with structured output
            transformer = LLMGraphTransformer(
                llm=self.llm,
                strict_mode=False,  # Don't enforce type restrictions
                ignore_tool_usage=False,  # Enable function calling for structured output
                node_properties=True,  # Extract node properties
                relationship_properties=True  # Extract relationship properties
            )
            
            # Process documents in batches and collect all graph documents
            all_graph_documents = []
            total_nodes = 0
            total_relationships = 0
            
            logger.info(f"Starting async graph extraction from {len(documents)} documents in batches of {batch_size}")
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(documents)-1)//batch_size + 1}")
                
                # Convert batch to graph documents using async method
                graph_documents = await transformer.aconvert_to_graph_documents(batch)
                
                # Log extraction details for this batch
                batch_nodes = sum(len(doc.nodes) for doc in graph_documents)
                batch_rels = sum(len(doc.relationships) for doc in graph_documents)
                total_nodes += batch_nodes
                total_relationships += batch_rels
                
                logger.info(f"Batch {i//batch_size + 1} extracted {batch_nodes} nodes and {batch_rels} relationships")
                
                all_graph_documents.extend(graph_documents)
            
            if not all_graph_documents:
                raise ValueError("No graph documents were generated from the input documents")
            
            logger.info(f"Completed extraction: {total_nodes} total nodes and {total_relationships} total relationships")
            
            # Initialize counters for successful insertions
            nodes_inserted = 0
            relationships_inserted = 0
            
            logger.info("Starting real-time graph construction and CSV writing")
            
            # Process each document's nodes and relationships immediately
            for doc_idx, graph_doc in enumerate(all_graph_documents, 1):
                logger.info(f"Processing document {doc_idx} of {len(all_graph_documents)}")
                
                # Insert nodes with immediate CSV writing
                if graph_doc.nodes:
                    for node in graph_doc.nodes:
                        self.insert_node({
                            'id': node.id,
                            'type': node.type,
                            'properties': node.properties
                        })
                        nodes_inserted += 1
                        if nodes_inserted % 10 == 0:  # Log progress every 10 nodes
                            logger.info(f"Inserted {nodes_inserted} nodes so far")
                
                # Insert relationships with immediate CSV writing
                if graph_doc.relationships:
                    for rel in graph_doc.relationships:
                        self.insert_relationship({
                            'source': {'id': rel.source.id},
                            'target': {'id': rel.target.id},
                            'type': rel.type,
                            'properties': rel.properties
                        })
                        relationships_inserted += 1
                        if relationships_inserted % 10 == 0:  # Log progress every 10 relationships
                            logger.info(f"Inserted {relationships_inserted} relationships so far")
            
            logger.info(f"Completed graph construction: {nodes_inserted} nodes and {relationships_inserted} relationships written to CSV")
            
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