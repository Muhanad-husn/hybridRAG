import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import logging

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM."""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
    ) -> None:
        self.llm = llm
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = not ignore_tool_usage
        
        # Initialize prompt
        self.prompt = (
            f"{system_prompt}\n\n"
            "Extract entities and relationships from the following text. "
            "Format your response as a list of JSON objects with 'head', 'head_type', "
            "'relation', 'tail', and 'tail_type' fields.\n\n"
            "Text: {input}"
        )
        
        # Add logging for initialization
        logger.info("Initializing LLMGraphTransformer")
        logger.info(f"Allowed nodes: {allowed_nodes}")
        logger.info(f"Allowed relationships: {allowed_relationships}")
        logger.info(f"Strict mode: {strict_mode}")
        logger.info(f"Function call mode: {self._function_call}")

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """Process a single document into a graph document."""
        try:
            text = document.page_content
            logger.info(f"Processing document: {document.metadata.get('source', 'unknown')}")
            logger.info(f"Document length: {len(text)} characters")
            
            # Format prompt
            formatted_prompt = self.prompt.format(input=text)
            
            # Get raw model response
            raw_response = self.llm.predict(formatted_prompt)
            
            # Log raw response
            logger.info("Raw model response:")
            logger.info("-" * 80)
            logger.info(raw_response)
            logger.info("-" * 80)
            
            # Extract nodes and relationships
            nodes = []
            relationships = []
            
            try:
                # Parse the response
                if isinstance(raw_response, str):
                    # Try to parse as JSON
                    parsed_json = json.loads(raw_response)
                    if isinstance(parsed_json, dict):
                        parsed_json = [parsed_json]
                else:
                    # Handle structured output
                    parsed_json = raw_response
                
                # Log parsed structure
                logger.info("Parsed structure:")
                logger.info(json.dumps(parsed_json, indent=2))
                
                # Extract nodes and relationships
                for item in parsed_json:
                    # Create nodes
                    head_node = Node(
                        id=item.get('head'),
                        type=item.get('head_type'),
                        properties={'source': document.metadata.get('source')}
                    )
                    tail_node = Node(
                        id=item.get('tail'),
                        type=item.get('tail_type'),
                        properties={'source': document.metadata.get('source')}
                    )
                    nodes.extend([head_node, tail_node])
                    
                    # Create relationship
                    relationship = Relationship(
                        source=head_node,
                        target=tail_node,
                        type=item.get('relation'),
                        properties={'source': document.metadata.get('source')}
                    )
                    relationships.append(relationship)
                
                # Log extraction results
                logger.info(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
                
            except Exception as e:
                logger.error(f"Error parsing model response: {str(e)}")
                logger.error("Falling back to empty graph")
                nodes = []
                relationships = []
            
            # Create graph document
            graph_doc = GraphDocument(
                nodes=nodes,
                relationships=relationships,
                source=document
            )
            
            return graph_doc
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents."""
        logger.info(f"Converting {len(documents)} documents to graph documents")
        results = []
        
        for doc in documents:
            try:
                graph_doc = self.process_response(doc, config)
                results.append(graph_doc)
            except Exception as e:
                logger.error(f"Error converting document: {str(e)}")
                # Continue with next document
                continue
        
        logger.info(f"Successfully converted {len(results)} documents")
        return results

    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """Asynchronously process a single document."""
        try:
            text = document.page_content
            logger.info(f"Async processing document: {document.metadata.get('source', 'unknown')}")
            
            # Format prompt
            formatted_prompt = self.prompt.format(input=text)
            
            # Get raw model response
            raw_response = await self.llm.apredict(formatted_prompt)
            
            # Log raw response
            logger.info("Raw model response (async):")
            logger.info("-" * 80)
            logger.info(raw_response)
            logger.info("-" * 80)
            
            # Extract nodes and relationships (similar to sync version)
            nodes = []
            relationships = []
            
            try:
                # Parse the response
                if isinstance(raw_response, str):
                    parsed_json = json.loads(raw_response)
                    if isinstance(parsed_json, dict):
                        parsed_json = [parsed_json]
                else:
                    parsed_json = raw_response
                
                # Extract nodes and relationships
                for item in parsed_json:
                    head_node = Node(
                        id=item.get('head'),
                        type=item.get('head_type'),
                        properties={'source': document.metadata.get('source')}
                    )
                    tail_node = Node(
                        id=item.get('tail'),
                        type=item.get('tail_type'),
                        properties={'source': document.metadata.get('source')}
                    )
                    nodes.extend([head_node, tail_node])
                    
                    relationship = Relationship(
                        source=head_node,
                        target=tail_node,
                        type=item.get('relation'),
                        properties={'source': document.metadata.get('source')}
                    )
                    relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"Error parsing async model response: {str(e)}")
                logger.error("Falling back to empty graph")
                nodes = []
                relationships = []
            
            # Create graph document
            graph_doc = GraphDocument(
                nodes=nodes,
                relationships=relationships,
                source=document
            )
            
            return graph_doc
            
        except Exception as e:
            logger.error(f"Error in async processing: {str(e)}")
            raise

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Asynchronously convert documents to graph documents."""
        logger.info(f"Async converting {len(documents)} documents")
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        logger.info(f"Successfully converted {len(results)} documents (async)")
        return results