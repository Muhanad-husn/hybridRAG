# SocioPolitics GraphMind: An Educational Tool for Enhanced Document Interaction

## Introduction

SocioPolitics GraphMind is an educational tool designed to assist researchers, journalists, and students in fields such as Politics, History, Sociology, Sociopolitical, and Socioeconomic studies. The app enables users to interact with documents through a chat interface, extracting structured insights and enhancing their research capabilities.

## Architecture Overview

SocioPolitics GraphMind uses HybridRAG which is built with a modular architecture, consisting of several layers:

- **Input Layer:** Handles document ingestion and segmentation.
  - [`src/input_layer/document_processor.py`](src/input_layer/document_processor.py)
- **Processing Layer:** Generates embeddings and constructs knowledge graphs.
  - [`src/processing_layer/graph_constructor.py`](src/processing_layer/graph_constructor.py)
  - [`src/tools/llm_graph_transformer.py`](src/tools/llm_graph_transformer.py)
- **Retrieval Layer:** Implements hybrid retrieval mechanisms.
  - [`Process_files.py`](Process_files.py)
- **Utility Components:** Manages API keys, configuration reloading, and caching.

## Key Functionalities and Endpoints

### Search Operations

- **Endpoint:** `/search`
- **Description:** Handles both Arabic and English queries, adjusts rerank counts, and integrates token count monitoring.
- **File:** [`app.py`](app.py)

### Document Processing

- **Endpoint:** `/process-documents`
- **Description:** Resets storage, processes documents, and builds vector stores and graphs.
- **File:** [`app.py`](app.py)

### File Management

- **Endpoints:** `/upload-files`, `/save-result`
- **Description:** Manages file uploads and result saving.
- **File:** [`app.py`](app.py)

### Configuration and Settings

- **Endpoints:** `/update-model-settings`, `/update-api-key`
- **Description:** Allows dynamic updates to model settings and API keys, providing user control.
- **File:** [`app.py`](app.py)

### System Monitoring

- **Endpoints:** `/logs`, `/get-document-node-counts`
- **Description:** Fetches logs and obtains document/node counts.
- **File:** [`app.py`](app.py)

## Benefits of the Technology & Design Choices

### Modular Architecture

- **Description:** Separation of concerns for document ingestion, processing, and retrieval allows flexible scalability.

### Robust Error Handling and Logging

- **Description:** Logging and error monitoring are applied at various endpoints to ensure reliability.

### User Control and Customization

- **Description:** Users can adjust settings for retrieval (`top_k`), model parameters (extraction/answer models, temperature, tokens), and API keys.

### Hybrid Retrieval Approach

- **Description:** Combines document processing with embeddings and knowledge graph construction to improve search relevancy and depth of information.
- **File:** [`src/tools/llm_graph_transformer.py`](src/tools/llm_graph_transformer.py)

### Scalability and Maintainability

- **Description:** Uses async operations for processing documents, caching strategies, and a modular design to simplify future extensions and maintenance.

## Key Features Overview

To delve deeper into specific features like hybrid retrieval, semantic chunking, or bilingual functionality, consult the [FEATURES.md](FEATURES.md) file. Here is a quick summary of some important highlights:

- **Hybrid Retrieval:** Combines keyword searches with semantic and graph-based approaches for robust performance and reduced missed connections.
- **Semantic Chunking:** Optimizes document segmentation, preserving context and improving retrieval accuracy.
- **Deduplication & Reranking:** Ensures result quality by removing duplicates and ordering the most relevant answers first.
- **Result Count Control:** Enables balancing coverage and efficiency within model token limits, adaptable to different LLMs.
- **Local Model Integration & Parallel Processing:** Executes embeddings, reranking, and translation locally to reduce costs and handle workloads more efficiently.
- **Bilingual Functionality:** Detects language automatically, defaults to generating English answers with Arabic translation, and provides an option to disable translation for faster performance.

## Detailed Walkthrough

### Flow Diagrams

- **Description:** Flow diagrams for search and document processing pipelines map the journey from file upload to result generation.

### Code Excerpts and Explanations

- **Description:** Annotated snippets from key files explaining business logic.
- **Files:** [`app.py`](app.py), [`Process_files.py`](Process_files.py)

### Configuration Files

- **Description:** Manages system settings and tuning.
- **File:** [`config/config.yaml`](config/config.yaml)

## Usage and User Guidance

### Quick Start

- **Instructions:** How to run the app (e.g., `python app.py` commands, setting up Docker if using [run_docker.txt](run_docker.txt)).

### Developer Guidelines

- **Description:** How to add new document processors or customize retrieval logic.

### Troubleshooting

- **Description:** Common issues with configurations, caching, and API responses.

## Future Enhancements and Roadmap

- **Description:** Planned improvements such as enhanced language support, additional document parsing modules, or integration with other data sources.
