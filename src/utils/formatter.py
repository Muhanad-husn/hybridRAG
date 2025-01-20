from typing import Dict, Any, Optional, Literal
import re
import os

OutputFormat = Literal["text", "markdown", "html"]

class ResultFormatter:
    """Enhanced formatter for search results with LLM-friendly output."""
    
    def __init__(self, format_type: OutputFormat = "text", max_length: int = 2000):
        """Initialize the formatter with desired output format and settings."""
        self.format_type = format_type
        self.max_length = max_length
        
    def _truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """Smartly truncate text at sentence boundary."""
        if not text:
            return ""
            
        max_len = max_length or self.max_length
        if len(text) <= max_len:
            return text
            
        # Try to truncate at the last sentence boundary
        truncated = text[:max_len]
        last_period = truncated.rfind('.')
        if last_period > max_len * 0.7:  # Only truncate at sentence if it's not too short
            truncated = truncated[:last_period + 1]
        else:
            # Truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
                
        return f"{truncated}..."
        
    def _format_score(self, score: float) -> str:
        """Format relevance score with appropriate scaling."""
        if score < 0.01:
            score = score * 100
        return f"{score * 100:.1f}%"

    def _clean_source_path(self, source: str) -> str:
        """Clean and format source path for display."""
        # Remove duplicate extensions
        source = re.sub(r'\.txt\.txt$', '.txt', source)
        # Get just the filename without path
        return os.path.basename(source)

    def _clean_line_numbers(self, text: str) -> str:
        """Remove line numbers from the beginning of lines."""
        return re.sub(r'^\d+\s*\|\s*', '', text, flags=re.MULTILINE)
        
    def format_document_result(self, doc: Any, score: float) -> str:
        """Format document search results."""
        content = self._clean_line_numbers(doc.page_content.strip())
        content = self._truncate_text(content)
        source = self._clean_source_path(doc.metadata.get('source', 'unknown'))
        
        sections = [
            "Context from document:",
            f"[{source} (Relevance: {self._format_score(score)})]",
            content,
            "---"
        ]
        
        return "\n".join(sections)
        
    def format_graph_result(self, result: Dict[str, Any]) -> str:
        """Format graph search results."""
        node = result['node']
        edges = result['edges']
        
        # Format edges with improved readability
        edge_info = []
        for edge in edges:
            edge_text = f"- {edge['type']} → {edge['target']}"
            edge_info.append(edge_text)
            
        edge_summary = "\n".join(edge_info) if edge_info else "No relationships found"
        
        sections = [
            "Graph Analysis:",
            f"Entity: {node['id']} (Type: {node['type']})",
            "Relationships:",
            edge_summary,
            "---"
        ]
        
        return "\n".join(sections)
        
    def format_reranked_result(self, result: Dict[str, Any]) -> str:
        """Format reranked search results."""
        # Get text content, handling both direct text and page_content
        text = result.get('text', '')
        if not text and 'page_content' in result:
            text = result.get('page_content', '')
        
        # Clean, remove line numbers, and truncate text
        if isinstance(text, str):
            text = self._clean_line_numbers(text.strip())
            text = self._truncate_text(text)
        else:
            return ""  # Return empty string if no valid text found
            
        # Get metadata, handling both direct meta and metadata
        meta = result.get('meta', '')
        if not meta and 'metadata' in result:
            meta = result.get('metadata', {}).get('source', 'unknown')
        meta = self._clean_source_path(str(meta))
        
        # Get score, defaulting to 0.0
        score = float(result.get('score', 0.0))
        
        # Only create output if we have valid text
        if not text.strip():
            return ""
            
        sections = [
            "Context from document:",
            f"[{meta} (Relevance: {self._format_score(score)})]",
            text,
            "---"
        ]
        
        return "\n".join(sections)

def format_result(result: Dict[str, Any], format_type: OutputFormat = "text") -> str:
    """Format a single search result for display."""
    formatter = ResultFormatter(format_type=format_type)
    
    if isinstance(result, tuple):
        # Handle document results from similarity search
        doc, score = result
        return formatter.format_document_result(doc, score)
    elif 'node' in result:
        # Handle graph search results
        return formatter.format_graph_result(result)
    else:
        # Handle reranked results
        return formatter.format_reranked_result(result)