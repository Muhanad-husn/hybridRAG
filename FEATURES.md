# Features

## Hybrid Retrieval

Hybrid retrieval is designed to combine traditional keyword search with semantic and graph-based methods.

- **Enhanced Accuracy:**  
  The system blends keyword precision with context-aware semantic matching. The core logic is executed in the [`retrieve_syria.py:run_hybrid_search`](retrieve_syria.py) routine, which also calls key components in [`src/retrieval_layer/hybrid_retrieval.py`](src/retrieval_layer/hybrid_retrieval.py) where relevance scoring and confidence calculation functions integrate multiple search paradigms.
  
- **Comprehensive Search:**  
  By constructing a document graph that connects chunks, the system captures explicit keywords, underlying concepts, and inter-document relationships, reducing missed connections during searches.
  
- **Robust Performance & Optimized Efficiency:**  
  The hybrid approach ensures handling of diverse query formats and language nuances while streamlining computation by balancing multiple retrieval methods.

## Semantic Chunking

Semantic chunking is about breaking large documents into smaller, coherent segments to preserve context. This not only optimizes processing but also improves retrieval accuracy:

- **Context Preservation:**  
  Document processors—such as those in [`src/input_layer/document_processor.py`](src/input_layer/document_processor.py)—manage the segmentation so that each chunk remains meaningfully intact.
  
- **Improved Processing:**  
  Focusing on contextually relevant chunks allows for more precise embedding and indexing rather than processing an entire document at once.
  
- **Refined Retrieval & Scalability:**  
  This chunking strategy ensures that search queries align with meaningful document sections, leading to faster retrieval, and scales effectively with large datasets.

## Deduplication and Reranking

Hybrid retrieval further enhances result quality by first applying a deduplication process. This mechanism filters out redundant entries by comparing content similarity and source information, leveraging caching and penalty strategies to ensure a diverse pool of unique documents.

Reranking is performed using a fine-tuned transformer model, which reassesses and reorders the initial search outcomes based on relevance. This dynamic process optimizes ranking thresholds according to query complexity, ensuring that the most accurate and valuable results are presented to the user.

Different specialized models are used for dense retrieval and for reranking. Dense retrieval quickly scans and retrieves a broad set of candidate documents using semantic embeddings, while the reranking model—fine-tuned for contextual understanding—precisely reorders results. This separation of concerns enhances both recall and precision, ensuring a robust and efficient search process.

## Result Count Control

The system provides combined control over the number of results, enabling users to define how much information is fed into answer generation. This ensures that the context remains within the model's token limits while preserving essential details. An automated mechanism dynamically adjusts the result count to balance comprehensive coverage with model efficiency. Furthermore, controlling the context length allows the process to serve different LLMs with varying token capacities. Users can tailor the amount of input and output data—catering from detailed, research-oriented queries to concise, quick responses—thus meeting the diverse needs of varying user categories and application scenarios.

## OpenRouter Integration & Model Customization

- **OpenRouter Integration:**
    OpenRouter provides a flexible interface that allows users to choose and customize models for entity extraction and answer generation, tailoring performance to their specific tasks. By default, it balances efficiency and cost using the following models:

        Extraction Model: google/gemini-flash-1.5 – A fast, cost-effective model optimized for high-frequency tasks, excelling in entity and relationship extraction from text and visual data.
        Answer Model: microsoft/phi-4 – A Free of charge reasoning-focused model trained on high-quality datasets, ensuring accurate, coherent responses with strong instruction adherence.

    Users can override these defaults, selecting models that best align with their needs, allowing for greater flexibility and adaptability in their workflows.

- **Model Flexibility:**  
  Users can swap out the default models with alternatives based on their needs, enabling tailored performance for specific tasks.

- **Controlled Model Temperature:**  
  The temperature parameter in GPT models controls the randomness and creativity of the generated output. Lower temperatures yield more predictable and deterministic responses, while higher temperatures produce more varied and creative outputs. Allowing users to adjust this parameter enables fine-tuning between predictability and innovation in responses. The temperature ranges from 0 to 1, with 0 producing the most deterministic outputs. As the temperature approaches 1, the likelihood of the model generating less coherent or "hallucinated" content increases.

## Local Model Integration & Parallel Processing

Several key models—including those for embedding generation, reranking, and translation—are executed locally. This design leverages patching techniques and parallel processing to reduce operational costs by avoiding expensive cloud-based APIs. While running these models on the local device omit cost, the processing time may vary based on the hardware capabilities. The embedding model (thenlper/gte-small), reranker (cross-encoder/ms-marco-MiniLM-L-12-v2), and translation models (e.g., MarianMT variants) are optimized to run concurrently, ensuring that the system can handle heavy workloads efficiently on multi-core devices, though performance will depend on available resources.

## Bilingual Functionality

The application is bilingual and automatically detects the user's language. By default, it generates responses in English first, then translates them into Arabic. Since the translation runs locally, longer responses may take additional time to process. To enhance performance, users can disable Arabic translation when not needed.
However, it's important to note that the system is optimized to handle English-language documents exclusively.

