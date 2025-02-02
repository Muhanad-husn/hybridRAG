# Hybrid RAG System Architecture

```mermaid
graph TB
    %% Data Sources
    DS[Data Sources] --> PP[Preprocessing Pipeline]
    
    %% Preprocessing
    PP --> VDB[(Vector Database)]
    PP --> KG[(Knowledge Graph)]
    
    %% Query Processing
    U[User Query] --> QP[Query Processor]
    QP --> |Vector Search| VDB
    QP --> |Graph Search| KG
    
    %% Hybrid Retrieval
    VDB --> HR[Hybrid Retriever]
    KG --> HR
    
    %% Response Generation
    HR --> RC[Reranking & Context]
    RC --> LLM[Large Language Model]
    LLM --> R[Final Response]

    %% Styling
    classDef database fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef input fill:#bfb,stroke:#333,stroke-width:2px
    classDef output fill:#fbb,stroke:#333,stroke-width:2px
    
    class VDB,KG database
    class PP,QP,HR,RC,LLM process
    class U,DS input
    class R output
```

## Component Description

1. **Data Sources**: Raw input data (documents, texts, structured data)
2. **Preprocessing Pipeline**: Handles document parsing, chunking, and embedding generation
3. **Vector Database**: Stores document embeddings for similarity search
4. **Knowledge Graph**: Stores semantic relationships between entities
5. **Query Processor**: Analyzes user queries and dispatches appropriate search strategies
6. **Hybrid Retriever**: Combines results from vector and graph-based searches
7. **Reranking & Context**: Prioritizes and formats retrieved information
8. **LLM**: Generates human-readable responses using retrieved context
