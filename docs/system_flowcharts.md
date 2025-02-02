# System Flowchart Documentation

## Mermaid Diagram

```mermaid
flowchart TD
    A[User Input] --> B[Document Processor]
    B --> C[Translator]
    C --> D[Embedding Generator]
    D --> E[Graph Constructor]
    E --> F[Hybrid Retrieval]
    F --> G[Output Results]
    
    subgraph Input Layer
        B
        C
    end
    
    subgraph Processing Layer
        D
        E
    end
    
    subgraph Retrieval Layer
        F
    end
    
    subgraph Utilities
        H[Logger]
        I[Error Handler]
    end
    
    B --> H
    C --> H
    D --> H
    E --> I
    F --> I
```

## Converting to Image

To convert this Mermaid diagram to an image, follow these steps:

1. Copy the entire Mermaid code block above (including the ```mermaid and ``` lines).
2. Go to the Mermaid Live Editor: https://mermaid.live/
3. Paste the copied code into the left panel of the editor.
4. The diagram will be rendered in the right panel.
5. Click on the "Download SVG" button in the top right corner to save the diagram as an image.
6. Optionally, you can also download as PNG or PDF using the respective buttons.

After downloading the image, you can add it to this document or use it in other documentation as needed.