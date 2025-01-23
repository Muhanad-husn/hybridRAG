import asyncio
import json
import string
import yaml
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from collections import defaultdict
import logging
import re
from thefuzz import fuzz

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
from .openrouter_client import OpenRouterClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = (
    "## Updated Knowledge Graph Instructions\n\n"
    "You are a **precise JSON generator** for knowledge graph construction. Your task is to:\n\n"
    "1. **Extract entities** and **relationships** from the provided text.\n"
    "2. **Format** them as **valid JSON objects**.\n"
    "3. **Preserve** exact multi-word entity names (e.g., \"Nelson Mandela\", \"United Nations\").\n"
    "4. **Ensure proper JSON syntax** with commas, quotes, and brackets.\n\n"
    "### Output Format\n\n"
    "- Return a **JSON array** of objects (`[...]`).\n"
    "- Each object must have the following **5 fields**:\n"
    "  1. `\"head\"`: Source entity (string)\n"
    "  2. `\"head_type\"`: Type of source entity (string)\n"
    "  3. `\"relation\"`: Relationship type in **UPPER_CASE** (string)\n"
    "  4. `\"tail\"`: Target entity (string)\n"
    "  5. `\"tail_type\"`: Type of target entity (string)\n\n"
    "#### Example\n\n"
    "[{\"head\":\"John\",\"head_type\":\"Person\",\"relation\":\"WORKS_FOR\",\"tail\":\"Microsoft\",\"tail_type\":\"Organization\"},"
    "{\"head\":\"Microsoft\",\"head_type\":\"Organization\",\"relation\":\"LOCATED_IN\",\"tail\":\"Seattle\",\"tail_type\":\"Location\"}]\n\n"
    "### Rules\n\n"
    "1. **Extract only explicitly stated information** (no assumptions or inferences).\n"
    "2. **Format your response as a SINGLE LINE** of valid JSON with **NO WHITESPACE** between properties, arrays, or objects.\n"
    "3. **Include commas between ALL properties** within an object and **between ALL objects** in the array.\n"
    "4. **Retain multi-word entities** exactly as they appear in the text (e.g., `\"The United Nations\"` → `\"head\":\"The United Nations\"`).\n"
    "5. **Each property must be in the format `\"key\":\"value\"`** with a colon between them (e.g., `\"relation\":\"LOCATED_IN\"`).\n\n"
    "### Example of Handling Multi-Word Entities\n\n"
    "If the text says:\n"
    "> \"Nelson Mandela led the African National Congress during the 1990s.\"\n\n"
    "You should produce something like:\n"
    "[{\"head\":\"Nelson Mandela\",\"head_type\":\"Person\",\"relation\":\"LEADS\",\"tail\":\"African National Congress\",\"tail_type\":\"Organization\"}]"
)

# Enhanced user prompt template
user_prompt_template = (
    "Extract entities and relationships from the following text as a SINGLE LINE of valid JSON array with NO WHITESPACE between properties:\n\n"
    "{text}\n\n"
    "Remember:\n"
    "1. Extract only explicitly stated information - no assumptions or inferences\n"
    "2. Preserve exact multi-word entities as they appear in the text\n"
    "3. Ensure valid JSON with commas between ALL properties and objects\n"
)

def extract_objects(text: str) -> List[Dict[str, str]]:
    """Extract objects from text using regex patterns."""
    objects = []
    # Match each object's content between curly braces
    for obj_match in re.finditer(r'{[^}]+}', text):
        obj_text = obj_match.group(0)
        # Extract key-value pairs
        pairs = {}
        
        # First pass: Look for standard key-value pairs with colons
        # Format: "key":"value" (including values with spaces)
        for pair_match in re.finditer(r'"(\w+)":"([^"]+)"', obj_text):
            key, value = pair_match.groups()
            pairs[key] = value
            
        # Second pass: Look for pairs with double quotes
        # Format: "key""value" (including values with spaces)
        if len(pairs) < 5:  # Only if we haven't found all pairs yet
            # First try to split by double quotes and colons
            parts = re.findall(r'"([^"]+)(?:":"|\s*"")([^"]+)"', obj_text)
            for key, value in parts:
                if key and value and key not in pairs:
                    pairs[key] = value
            
            # If still missing pairs, try more aggressive parsing
            if len(pairs) < 5:
                # Remove curly braces and split by double quotes
                clean_text = obj_text.strip('{}')
                parts = clean_text.split('"')
                # Filter out empty strings and process pairs
                parts = [p for p in parts if p and not p.isspace()]
                for i in range(0, len(parts)-1, 2):
                    if i+1 < len(parts):
                        key = parts[i]
                        value = parts[i+1]
                        if key and value and key not in pairs:
                            pairs[key] = value
        
        # If we found all required fields, add the object
        if all(k in pairs for k in ['head', 'head_type', 'relation', 'tail', 'tail_type']):
            objects.append(pairs)
            logger.debug(f"Extracted object: {pairs}")
        else:
            logger.warning(f"Skipping incomplete object, found keys: {list(pairs.keys())}")
    
    return objects

def repair_json(text: str) -> str:
    """Repair common JSON formatting issues."""
    # Extract just the JSON array part
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in text")
    text = text[start:end]
    
    # Remove all whitespace except within quotes
    text = re.sub(r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', '', text)
    
    # First pass: Fix property separators
    # Convert "head""head_type" to "head","head_type"
    text = re.sub(r'"([^"]+)""([^"]+)"', r'"\1","\2"', text)
    
    # Second pass: Fix property-value pairs
    # Convert "head_type""Event" to "head_type":"Event"
    text = re.sub(r'"(\w+)"(?!:)"([^"]+)"', r'"\1":"\2"', text)
    
    # Third pass: Fix any remaining issues
    # Add missing commas between objects
    text = re.sub(r'}{', '},{', text)
    # Remove any extra commas at the start of objects
    text = re.sub(r'{,', '{', text)
    # Remove any trailing commas before closing brackets
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    try:
        # Try to parse the JSON
        data = json.loads(text)
        return json.dumps(data, indent=4)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Problematic JSON: {text}")
        # If parsing fails, try to extract objects manually
        objects = extract_objects(text)
        if objects:
            return json.dumps(objects, indent=4)
        else:
            raise ValueError("Failed to parse JSON")

def normalize_label(label: str) -> str:
    """Normalize entity labels for comparison.
    
    Args:
        label: The entity label to normalize
        
    Returns:
        Normalized version of the label (lowercase, no punctuation, trimmed)
    """
    # Lowercase
    label = label.lower()
    # Remove punctuation
    label = label.translate(str.maketrans('', '', string.punctuation))
    # Trim whitespace
    label = label.strip()
    return label

def find_duplicate_nodes(nodes: List[Dict[str, Any]], similarity_threshold: int = 80) -> List[Tuple[int, int]]:
    """Find potential duplicate nodes using fuzzy matching.
    
    Args:
        nodes: List of node dictionaries with 'id' and 'type' keys
        similarity_threshold: Minimum similarity score to consider nodes as duplicates
        
    Returns:
        List of tuples containing indices of duplicate node pairs
    """
    # Group nodes by type and first 3 letters of normalized label
    blocks = defaultdict(list)
    for idx, node in enumerate(nodes):
        normalized = normalize_label(node['id'])
        block_key = (node['type'], normalized[:3])
        blocks[block_key].append(idx)
    
    # Find duplicates within each block using fuzzy matching
    duplicates = []
    for block_indices in blocks.values():
        n = len(block_indices)
        if n < 2:
            continue
            
        for i in range(n):
            for j in range(i + 1, n):
                idx_a = block_indices[i]
                idx_b = block_indices[j]
                
                label_a = normalize_label(nodes[idx_a]['id'])
                label_b = normalize_label(nodes[idx_b]['id'])
                
                similarity = fuzz.ratio(label_a, label_b)
                if similarity >= similarity_threshold:
                    duplicates.append((idx_a, idx_b))
    
    return duplicates

def merge_duplicate_nodes(nodes: List[Dict[str, Any]], relationships: List[Relationship]) -> Tuple[List[Dict[str, Any]], List[Relationship]]:
    """Merge duplicate nodes and update relationships accordingly.
    
    Args:
        nodes: List of node dictionaries
        relationships: List of relationships between nodes
        
    Returns:
        Tuple of (deduplicated nodes list, updated relationships list)
    """
    # Find duplicates
    duplicates = find_duplicate_nodes(nodes)
    if not duplicates:
        return nodes, relationships
        
    # Create merge mapping
    merge_map = {}  # old_id -> canonical_id
    for idx_a, idx_b in duplicates:
        # Use the first occurrence as canonical
        canonical_node = nodes[idx_a]
        merged_node = nodes[idx_b]
        merge_map[merged_node['id']] = canonical_node['id']
        
        # Merge properties if they exist
        if 'properties' in merged_node and 'properties' in canonical_node:
            for key, value in merged_node['properties'].items():
                if key not in canonical_node['properties']:
                    canonical_node['properties'][key] = value
    
    # Update relationships
    updated_relationships = []
    for rel in relationships:
        source_id = rel.source.id
        target_id = rel.target.id
        
        # Update source and target if they were merged
        if source_id in merge_map:
            rel.source.id = merge_map[source_id]
        if target_id in merge_map:
            rel.target.id = merge_map[target_id]
            
        updated_relationships.append(rel)
    
    # Create final deduplicated node list
    canonical_nodes = {}
    for node in nodes:
        node_id = node['id']
        if node_id in merge_map:
            continue  # Skip merged nodes
        canonical_nodes[node_id] = node
    
    return list(canonical_nodes.values()), updated_relationships

def process_json_response(raw_response: str) -> List[Dict[str, str]]:
    """Process raw JSON response from LLM."""
    # First try to extract objects directly
    objects = extract_objects(raw_response)
    if objects:
        # Add commas between properties if missing
        processed_objects = []
        for obj in objects:
            processed_obj = {}
            for key, value in obj.items():
                # Clean up any remaining formatting issues
                key = key.strip().strip('"')
                value = value.strip().strip('"')
                processed_obj[key] = value
            if all(k in processed_obj for k in ['head', 'head_type', 'relation', 'tail', 'tail_type']):
                processed_objects.append(processed_obj)
                logger.debug(f"Processed object: {processed_obj}")
        
        if processed_objects:
            logger.debug(f"Successfully processed {len(processed_objects)} objects")
            return processed_objects
    
    # If direct extraction fails, try repairing the JSON
    try:
        repaired_json = repair_json(raw_response)
        parsed_json = json.loads(repaired_json)
        if isinstance(parsed_json, dict):
            logger.debug("Parsed single JSON object")
            return [parsed_json]
        logger.debug(f"Parsed {len(parsed_json)} JSON objects")
        return parsed_json
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        return []

class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM."""
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        config_path: str = "config/config.yaml"
    ) -> None:
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize OpenRouter client with extraction model
        self.llm = OpenRouterClient(model=self.config['llm']['extraction_model'])
        
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = False  # Always disable strict mode to allow all types
        self._function_call = not ignore_tool_usage
        
        # Add logging for initialization
        logger.info("Initializing LLMGraphTransformer")
        logger.info(f"Using extraction model: {self.config['llm']['extraction_model']}")
        logger.info(f"Allowed nodes: {allowed_nodes}")
        logger.info(f"Allowed relationships: {allowed_relationships}")
        logger.info(f"Strict mode: {self.strict_mode}")
        logger.info(f"Function call mode: {self._function_call}")

    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """Asynchronously process a single document."""
        try:
            text = document.page_content
            #logger.info(f"Async processing document: {document.metadata.get('source', 'unknown')}")
            
            # Get raw model response using predict
            if hasattr(self.llm, 'client'):
                # Note: OpenRouter client doesn't have async methods yet
                response = self.llm.get_completion(
                    prompt=user_prompt_template.format(text=text),
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=2000
                )
                raw_response = response.get('content', '')
            else:
                # Fallback to standard apredict
                raw_response = await self.llm.apredict(text=f"{system_prompt}\n\n{user_prompt_template.format(text=text)}")
            
            # Log raw response
            #logger.info("Raw model response (async):")
            #logger.info("-" * 80)
            #logger.info(raw_response)
            #logger.info("-" * 80)
            
            # Extract nodes and relationships (similar to sync version)
            nodes = []
            relationships = []
            node_map = {}  # Keep track of created nodes to avoid duplicates
            
            try:
                # Process the JSON response
                objects = process_json_response(raw_response)
                
                # Extract nodes and relationships
                for item in objects:
                    # Skip if required fields are missing
                    if not all(k in item for k in ['head', 'head_type', 'relation', 'tail', 'tail_type']):
                        logger.warning(f"Skipping incomplete object: {item}")
                        continue
                    
                    # Create or get head node
                    head_id = item['head']
                    if head_id not in node_map:
                        head_node = Node(
                            id=head_id,
                            type=item['head_type'],
                            properties={'source': document.metadata.get('source')}
                        )
                        nodes.append(head_node)
                        node_map[head_id] = head_node
                        logger.debug(f"Created head node: {head_id}")
                    else:
                        head_node = node_map[head_id]
                        logger.debug(f"Using existing head node: {head_id}")
                    
                    # Create or get tail node
                    tail_id = item['tail']
                    if tail_id not in node_map:
                        tail_node = Node(
                            id=tail_id,
                            type=item['tail_type'],
                            properties={'source': document.metadata.get('source')}
                        )
                        nodes.append(tail_node)
                        node_map[tail_id] = tail_node
                        logger.debug(f"Created tail node: {tail_id}")
                    else:
                        tail_node = node_map[tail_id]
                        logger.debug(f"Using existing tail node: {tail_id}")
                    
                    # Create relationship
                    relationship = Relationship(
                        source=head_node,
                        target=tail_node,
                        type=item['relation'],
                        properties={'source': document.metadata.get('source')}
                    )
                    relationships.append(relationship)
                    logger.debug(f"Created relationship: {head_id} -{item['relation']}-> {tail_id}")
                
            except Exception as e:
                logger.error(f"Error parsing async model response: {str(e)}")
                logger.error("Falling back to empty graph")
                nodes = []
                relationships = []
            
            # Perform node deduplication
            if nodes and relationships:
                #logger.info("Starting node deduplication...")
                # Convert Node objects to dictionaries for deduplication
                node_dicts = [{'id': node.id, 'type': node.type, 'properties': node.properties} for node in nodes]
                
                # Deduplicate nodes and update relationships
                deduplicated_node_dicts, updated_relationships = merge_duplicate_nodes(node_dicts, relationships)
                
                # Convert back to Node objects
                deduplicated_nodes = [
                    Node(
                        id=node_dict['id'],
                        type=node_dict['type'],
                        properties=node_dict['properties']
                    )
                    for node_dict in deduplicated_node_dicts
                ]
                
                logger.info(f"Deduplication complete. Original nodes: {len(nodes)}, Deduplicated nodes: {len(deduplicated_nodes)}")
                
                # Create graph document with deduplicated nodes
                graph_doc = GraphDocument(
                    nodes=deduplicated_nodes,
                    relationships=updated_relationships,
                    source=document
                )
            else:
                # Create empty graph document if no nodes/relationships
                graph_doc = GraphDocument(
                    nodes=[],
                    relationships=[],
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