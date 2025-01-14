
# Instruction Card for Document Processing

**Objective**: Implement document ingestion and segmentation by iterating over all files in a folder. Support **PDF**, **DOCX**, and **TXT** file formats using LLMSherpa, with the same parameter values as the provided sample code.

---

## Instructions

1. **Folder Iteration**:
   - Implement logic to iterate over all files in a specified folder.
   - Filter files to include only those with **PDF**, **DOCX**, and **TXT** extensions.

2. **Tool/Library to Use**:
   - Use the `LLMSherpaFileLoader` as demonstrated in the `LLMSherpa_langchain_example.py` file.

3. **Parameter Values**:
   - Configure `LLMSherpaFileLoader` with the **same parameter values** as shown in the provided sample code in `LLMSherpa_langchain_example.py`. 

4. **File Path Handling**:
   - Dynamically pass each file path from the folder to `LLMSherpaFileLoader`.

5. **Output**:
   - Store the parsed content of each file as a list of chunks.
   - Maintain a dictionary or list to track files and their corresponding processed content.

6. **Error Handling**:
   - Skip unsupported files with a warning in the logs.
   - Handle exceptions during file loading (e.g., failed OCR or unsupported format) gracefully.

7. **Testing**:
   - Test with a folder containing mixed file types to ensure only supported files are processed.

---

**Reference**: `LLMSherpa_langchain_example.py` file in the project code-base.

# Instruction Card for Embedding Generation

**Objective**: Generate embeddings for document chunks using an offline-compatible embedding model and store them for retrieval.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the embedding model demonstrated in `gte_small_embedding_example.py`.

2. **Parameter Values**:
   - Configure the model, tokenizer, and inference mode as shown in the sample code in `gte_small_embedding_example.py`.

3. **Input**:
   - The input will consist of document chunks output from the document processing step.

4. **Output**:
   - Store the embeddings as fixed-size vectors corresponding to each chunk. Ensure they are in a format compatible with the storage mechanism (e.g., ArangoDB).

5. **Storage**:
   - Use the storage mechanism provided by the `vector_store_index.py` file to store embeddings locally.

6. **Error Handling**:
   - Implement exception handling for missing or malformed chunks.

7. **Testing**:
   - Validate embeddings by checking their dimensions and inspecting similarity scores for example queries.

---

**Reference**: 
- `gte_small_embedding_example.py` for embedding generation.
- `vector_store_index.py` for embedding storage.

# Instruction Card for Knowledge Graph Construction

**Objective**: Create a knowledge graph by extracting entities and relationships from document chunks and organizing them into a structured graph database.

---

## Instructions

1. **Tool/Library to Use**:
   - Refer to the `llm_graph_transformer.py` file for implementing the knowledge graph construction process.

2. **Entity and Relationship Extraction**:
   - Use the extraction schema for nodes and relationships as demonstrated in `llm_graph_transformer.py`.
   - Ensure adherence to the schema rules for entity consistency and relationship generalization.

3. **Graph Schema**:
   - Use the node and relationship types specified in the provided schema of `llm_graph_transformer.py`.

4. **Database Integration**:
   - Store the extracted entities and relationships in the graph database.
   - Refer to the database setup and insertion methods in `arangodb_graph_setup.py` for guidance.

5. **Input**:
   - Processed document chunks from earlier steps will serve as input for entity and relationship extraction.

6. **Output**:
   - A knowledge graph stored in the database, with nodes and relationships adhering to the predefined schema.

7. **Error Handling**:
   - Implement validation to ensure extracted entities and relationships conform to the expected schema.
   - Handle missing or malformed input data gracefully.

8. **Testing**:
   - Test the graph construction with a variety of documents to ensure proper extraction and storage of entities and relationships.

---

**References**:
- `llm_graph_transformer.py` for entity and relationship extraction.
- `arangodb_graph_setup.py` for graph database setup and data insertion.

# Instruction Card for Hybrid Retrieval

**Objective**: Implement a hybrid retrieval system that combines graph-based and embedding-based search results, incorporating a fallback hierarchy.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the retrieval and ranking methods as demonstrated in `RAG_with_Ranking_and_Reranking.py` and `offline_graph_rag_tool.py`.

2. **Combination Logic**:
   - Implement logic to combine results from graph-based and embedding-based retrieval methods.
   - Incorporate a fallback hierarchy where graph-based retrieval results take precedence, and embedding-based results serve as a backup.

3. **Graph-Based Retrieval**:
   - Use graph traversal techniques to retrieve nodes and relationships from the knowledge graph.
   - Refer to `offline_graph_rag_tool.py` for integration of graph-based retrieval methods.

4. **Embedding-Based Retrieval**:
   - Use the embedding similarity-based search demonstrated in `RAG_with_Ranking_and_Reranking.py` to retrieve relevant documents.

5. **Ranking and Re-Ranking**:
   - Utilize the ranking and reranking techniques provided in `RAG_with_Ranking_and_Reranking.py` to prioritize the most relevant results.

6. **Input**:
   - Queries and the stored knowledge graph and embeddings from earlier steps will serve as input.

7. **Output**:
   - Combined ranked results from graph-based and embedding-based retrieval, adhering to the defined fallback hierarchy.

8. **Error Handling**:
   - Ensure robustness when either graph-based or embedding-based retrieval fails.
   - Log errors and fallback to the available retrieval method.

9. **Testing**:
    - Validate the retrieval system with queries of varying complexity to ensure effective hybrid retrieval and fallback functionality.

10. **LLM Processing**:
    - Process retrieved results with an LLM (using OpenRouter) to generate answers
    - Format retrieved context into a clear prompt for the LLM
    - Include system instructions to ensure answers are based only on provided context
    - Handle LLM API errors gracefully with appropriate fallback behavior
    - Return both the LLM-generated answer and supporting context (optional display)

---

**References**:
- `RAG_with_Ranking_and_Reranking.py` for embedding-based retrieval and ranking.
- `offline_graph_rag_tool.py` for graph-based retrieval integration.

# Instruction Card for Data Deduplication

**Objective**: Identify and remove duplicate results during the retrieval and ranking stages to ensure output clarity and relevance.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the deduplication logic demonstrated in `offline_graph_rag_tool.py`.

2. **Duplication Criteria**:
   - Define duplication based on:
     - Identical content in retrieved documents.
     - Overlapping metadata or similar sources.

3. **Implementation Steps**:
   - Parse the combined results from the retrieval stage (both graph-based and embedding-based).
   - Use a hash-based or tuple-based mechanism (as shown in `offline_graph_rag_tool.py`) to detect duplicates:
     - Hash or tuple = `(document content, metadata source)`.
   - Compare hashes or tuples to identify duplicates.
   - Retain only unique results in the final output.

4. **Handling Similar Results**:
   - For documents with high semantic similarity but minor differences, rank them based on relevance scores and retain the most relevant one.
   - Leverage similarity metrics (e.g., cosine similarity) to flag near-duplicates.

5. **Integration**:
   - Integrate deduplication as a post-processing step in the hybrid retrieval pipeline.
   - Ensure deduplication is applied before reranking to optimize performance.

6. **Input**:
   - Combined retrieval results from graph-based and embedding-based methods.

7. **Output**:
   - A deduplicated list of results, ready for ranking and reranking.

8. **Error Handling**:
   - Log errors during comparison (e.g., malformed data).
   - Use a fallback mechanism to return partial deduplicated results if processing fails.

9. **Testing**:
   - Validate deduplication by testing with:
     - Results containing duplicates (exact and near-duplicates).
     - Large datasets to ensure scalability.
   - Check the integrity of retained results (no missing or altered documents).

---

**References**:
- `offline_graph_rag_tool.py` for deduplication logic and implementation examples.

# Instruction Card for Parallel Processing

**Objective**: Enable parallel processing to handle multiple documents or queries simultaneously, improving the system's efficiency.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the `concurrent.futures.ThreadPoolExecutor` or `multiprocessing` module as demonstrated in `offline_graph_rag_tool.py`.

2. **Concurrency Model**:
   - Choose between threading or multiprocessing based on:
     - **Threading**: Use for I/O-bound tasks (e.g., file reading or API calls).
     - **Multiprocessing**: Use for CPU-bound tasks (e.g., embedding generation, graph construction).

3. **Implementation Steps**:
   - Divide the input workload (e.g., documents or queries) into smaller batches.
   - Implement parallel processing to process batches concurrently:
     - Use `ThreadPoolExecutor` for tasks requiring concurrent network or file operations.
     - Use `multiprocessing.Pool` for CPU-intensive tasks like embedding generation.

4. **Batch Size and Workers**:
   - Determine optimal batch size and the number of worker threads or processes based on:
     - Hardware capabilities (e.g., CPU cores, RAM).
     - Task complexity and input size.

5. **Error Handling**:
   - Implement robust error handling within the parallel processing loop:
     - Catch and log exceptions for individual tasks without halting the entire process.
     - Retry failed tasks or return partial results where possible.

6. **Integration**:
   - Integrate parallel processing within the pipeline stages requiring high throughput:
     - Document processing.
     - Embedding generation.
     - Knowledge graph construction.
     - Retrieval and ranking.

7. **Input**:
   - Input workload (e.g., a list of documents or queries) divided into batches for parallel execution.

8. **Output**:
   - Processed results from all tasks combined into a single output structure.

9. **Testing**:
   - Test parallel processing with varying batch sizes and worker counts to ensure scalability.
   - Validate correctness by comparing outputs with those from sequential execution.

---

**References**:
- `offline_graph_rag_tool.py` for examples of parallel processing with `ThreadPoolExecutor`.

# Instruction Card for Testing and Validation

**Objective**: Validate the individual and integrated modules to ensure proper functionality, robustness, and seamless operation.

---

## Instructions

1. **Tool/Library to Use**:
   - Utilize logging mechanisms provided in `logger.py` for detailed test logging and reporting.

2. **Test Cases**:
   - **Module-Level Testing**:
     - Test document processing with various file formats (PDF, DOCX, TXT) using the configurations in `LLMSherpa_langchain_example.py`.
     - Validate embedding generation using `gte_small_embedding_example.py` by ensuring proper vector dimensions and similarity scores.
     - Check knowledge graph construction with `llm_graph_transformer.py` for correct entity and relationship mapping.
     - Test hybrid retrieval combining graph-based and embedding-based methods using `offline_graph_rag_tool.py` and `RAG_with_Ranking_and_Reranking.py`.
   - **Integration Testing**:
     - Simulate end-to-end workflows, from document ingestion to result retrieval.
     - Ensure consistency in data flow and expected outputs across all modules.

3. **Input**:
   - Use a diverse dataset of test files, including edge cases such as incomplete or malformed documents.

4. **Output**:
   - Generate logs with clear success/failure metrics for each module and integration test.
   - Create detailed reports summarizing test outcomes.

5. **Error Handling**:
   - Verify the robustness of error handling mechanisms implemented in each module.
   - Simulate error scenarios (e.g., missing files, malformed inputs) and ensure the system recovers gracefully.

6. **Performance Metrics**:
   - Measure system performance, including:
     - Time taken for document processing, embedding generation, and retrieval.
     - Memory and CPU utilization for parallel processing workflows.

7. **Testing Automation**:
   - Automate tests using Python’s `unittest` or `pytest` frameworks to enable continuous validation.

8. **Validation Criteria**:
   - Ensure that all modules meet predefined functional requirements.
   - Verify that hybrid retrieval outputs consistent and relevant results.

---

**References**:
- `logger.py` for logging and reporting.
- `LLMSherpa_langchain_example.py` for document processing tests.
- `gte_small_embedding_example.py` for embedding validation.
- `llm_graph_transformer.py` for knowledge graph validation.
- `offline_graph_rag_tool.py` and `RAG_with_Ranking_and_Reranking.py` for hybrid retrieval tests.

# Instruction Card for Error Handling and Logging

**Objective**: Implement robust error recovery and logging mechanisms for all modules to ensure smooth operation and efficient debugging.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the `logging` module and configuration provided in `logger.py` for consistent logging across all modules.

2. **Error Detection**:
   - Monitor each module for common errors:
     - Document ingestion errors (e.g., unsupported file types, missing files).
     - Embedding generation failures (e.g., invalid input dimensions).
     - Knowledge graph construction issues (e.g., schema mismatches, database connectivity problems).
     - Retrieval and ranking errors (e.g., missing or malformed queries).

3. **Error Logging**:
   - Log errors with detailed messages, including:
     - Module and function name where the error occurred.
     - Relevant input data that caused the error.
     - Timestamp and stack trace for debugging.
   - Use different log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) for appropriate categorization.

4. **Recovery Mechanisms**:
   - Implement fallback mechanisms to ensure partial results in case of errors:
     - Skip problematic files or queries and continue processing the rest.
     - Default to simpler retrieval methods if advanced methods fail.
   - Log fallback actions taken for traceability.

5. **Timeout Handling**:
   - Use timeout mechanisms for tasks that may hang (e.g., long retrieval queries).
   - Refer to the timeout decorator in `offline_graph_rag_tool.py` for implementation.

6. **Integration with Logging**:
   - Integrate logging into all modules:
     - Document processing: Log successful and failed file ingestion.
     - Embedding generation: Log successful embedding storage and anomalies.
     - Knowledge graph construction: Log entity and relationship extraction summaries.
     - Retrieval and ranking: Log query execution and results.

7. **Output**:
   - Generate log files for each execution, categorized by modules.
   - Maintain a rotating log system as configured in `logger.py` to manage log file size.

8. **Testing**:
   - Simulate common errors (e.g., invalid inputs, database disconnections) to verify logging accuracy and recovery mechanisms.
   - Ensure no critical errors halt the entire system.

---

**References**:
- `logger.py` for logging setup and configurations.
- `offline_graph_rag_tool.py` for timeout handling and error detection mechanisms.

# Instruction Card for Timeout Management

**Objective**: Implement time limits for processing requests and provide fallback or partial results when timeouts occur.

---

## Instructions

1. **Tool/Library to Use**:
   - Use the timeout decorator available in `offline_graph_rag_tool.py`.

2. **Timeout Implementation**:
   - Define maximum allowed execution time for critical operations:
     - Document processing.
     - Embedding generation.
     - Knowledge graph construction.
     - Retrieval and ranking queries.
   - Wrap these operations with the timeout decorator to ensure they terminate if the time limit is exceeded.

3. **Fallback Mechanisms**:
   - Return partial results or meaningful error messages when a timeout occurs:
     - For document processing, process completed files and skip remaining ones.
     - For embedding generation or retrieval, return results available up to the point of timeout.

4. **Timeout Configuration**:
   - Configure timeout durations based on expected operation complexity and hardware limitations:
     - Example: 10 seconds for document ingestion, 20 seconds for retrieval tasks.
   - Make timeout values configurable via an external settings file for flexibility.

5. **Integration with Error Handling**:
   - Log timeout events using the logging mechanisms in `logger.py`.
   - Ensure error logs include the operation name, duration, and any partial results returned.

6. **Testing**:
   - Simulate scenarios with heavy workloads or delays to trigger timeouts.
   - Validate that operations terminate gracefully and fallback mechanisms provide usable outputs.

7. **Output**:
   - Ensure the system remains responsive under load by terminating stalled processes.
   - Provide logs with clear descriptions of timeout occurrences and actions taken.

---

**References**:
- `offline_graph_rag_tool.py` for the timeout decorator.
- `logger.py` for logging timeout events.
