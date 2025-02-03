# SocioPolitics GraphMind: An Educational Tool for Enhanced Document Interaction

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Key Functionalities and Endpoints](#key-functionalities-and-endpoints)
   - [Search Operations](#search-operations)
   - [Document Processing](#document-processing)
   - [File Management](#file-management)
   - [Configuration and Settings](#configuration-and-settings)
   - [System Monitoring](#system-monitoring)
4. [Benefits of the Technology & Design Choices](#benefits-of-the-technology--design-choices)
   - [Modular Architecture](#modular-architecture)
   - [Robust Error Handling and Logging](#robust-error-handling-and-logging)
   - [User Control and Customization](#user-control-and-customization)
   - [Hybrid Retrieval Approach](#hybrid-retrieval-approach)
   - [Scalability and Maintainability](#scalability-and-maintainability)
5. [Expanded Feature Highlights](#expanded-feature-highlights)
   - [1. Hybrid Retrieval](#1-hybrid-retrieval)
   - [2. Semantic Chunking](#2-semantic-chunking)
   - [3. Deduplication & Reranking](#3-deduplication--reranking)
   - [4. Result Count Control](#4-result-count-control)
   - [5. OpenRouter Integration & Model Customization](#5-openrouter-integration--model-customization)
   - [6. Local Model Integration & Parallel Processing](#6-local-model-integration--parallel-processing)
   - [7. Bilingual Functionality](#7-bilingual-functionality)
6. [Usage and User Guidance](#usage-and-user-guidance)
   - [0. Obtain an OpenRouter API Key (If Required)](#0-obtain-an-openrouter-api-key-if-required)
   - [1. Clone the Repository](#1-clone-the-repository)
   - [2. Create-a-Virtual-Environment-recommended](#2-create-a-virtual-environment-recommended)
   - [3. Install Requirements](#3-install-requirements)
   - [4. Run the Application](#4-run-the-application)

---

## Introduction

**SocioPolitics GraphMind** is an educational tool designed to assist researchers, journalists, and students in disciplines like Politics, History, Sociology, and Socioeconomic studies. It enables users to interact with their documents through a chat interface, extracting structured insights and boosting research potential.

---

## Architecture Overview

SocioPolitics GraphMind uses a **Hybrid Retrieval** architecture (sometimes referred to as **HybridRAG**), with a modular design that breaks down into several layers:

- **Input Layer**: Handles document ingestion and segmentation.\
  *Key File:* `src/input_layer/document_processor.py`

- **Processing Layer**: Generates embeddings, constructs knowledge graphs, and transforms these structures for further analysis.\
  *Key Files:*\

  - `src/processing_layer/graph_constructor.py`\

  - `src/tools/llm_graph_transformer.py`

- **Retrieval Layer**: Implements hybrid retrieval logic, blending keyword searches with semantic and graph-based methods.\
  *Key Files:*\

  - `Process_files.py`\

  - `retrieve_syria.py`\

  - `src/retrieval_layer/hybrid_retrieval.py`

- **Utility Components**: Manages API keys, configuration reloading, and caching.

---

## Key Functionalities and Endpoints

### Search Operations

- **Endpoint**: `/search`\

- **Description**: Accepts Arabic and English queries, applies semantic chunking, deduplication, and reranking. Monitors token usage to stay within model limits.\

- **File**: `app.py`

### Document Processing

- **Endpoint**: `/process-documents`\

- **Description**: Resets storage, processes uploaded documents, and builds vector stores and knowledge graphs.\

- **File**: `app.py`

### File Management

- **Endpoints**: `/upload-files`, `/save-result`\

- **Description**: Manages file uploads and persists processed outputs.\

- **File**: `app.py`

### Configuration and Settings

- **Endpoints**: `/update-model-settings`, `/update-api-key`\

- **Description**: Dynamically updates model preferences (extraction/answer models, temperature) and API keys, ensuring flexibility for diverse use cases.\

- **File**: `app.py`

### System Monitoring

- **Endpoints**: `/logs`, `/get-document-node-counts`\

- **Description**: Provides logs and high-level document-node statistics for oversight and debugging.\

- **File**: `app.py`

---

## Benefits of the Technology & Design Choices

### Modular Architecture

Separating concerns into distinct modules (document ingestion, processing, retrieval) makes the system easy to scale and maintain.

### Robust Error Handling and Logging

Comprehensive logging and error monitoring at multiple endpoints enhance stability and help diagnose issues efficiently.

### User Control and Customization

By exposing endpoints to modify retrieval parameters (`top_k`), model settings (extraction/answer model, temperature), and API keys, the system empowers users to tailor performance and cost.

### Hybrid Retrieval Approach

Combines document processing, semantic embeddings, and a knowledge graph to improve search relevancy and depth. This multi-pronged strategy reduces missed connections and boosts retrieval accuracy.

### Scalability and Maintainability

Async document processing, caching strategies, and a modular structure help the system handle bigger datasets without sacrificing performance.

---

## Expanded Feature Highlights

### 1. Hybrid Retrieval

- **Enhanced Accuracy**\
  Blends traditional keyword matching with semantic embeddings and a knowledge graph for deeper context. Main logic resides in the `retrieve_syria.py:run_hybrid_search` routine and `src/retrieval_layer/hybrid_retrieval.py`.

- **Comprehensive Search**\
  Builds a document graph that connects text chunks, capturing explicit keywords, underlying concepts, and cross-document relationships.

- **Robust Performance & Efficiency**\
  Optimizes search by leveraging multiple methods to handle diverse queries in different languages or styles.

### 2. Semantic Chunking

- **Context Preservation**\
  Large documents are broken into coherent segments, preserving context in each chunk. Implemented within `src/input_layer/document_processor.py`.

- **Improved Processing**\
  Processing contextually relevant chunks (instead of entire documents) yields better embedding quality and indexing precision.

- **Refined Retrieval & Scalability**\
  Each chunk aligns closely with a query, improving accuracy and enabling the system to scale across extensive corpora.

### 3. Deduplication & Reranking

- **Deduplication**\
  The system compares similarity and origin to filter out redundant entries, providing a diverse set of relevant documents.

- **Reranking**\
  A fine-tuned transformer model then reorders the remaining candidates, ensuring that the most relevant results rise to the top.

- **Separation of Concerns**\
  Dense retrieval retrieves broad candidates efficiently, while the reranker (cross-encoder) re-sorts them for higher precision.

### 4. Result Count Control

- **Dynamic Control**\
  Users can set how many results feed into answer generation, maintaining a balance between comprehensive coverage and token limits.

- **Adaptive Mechanism**\
  Automatically adjusts to accommodate different LLMs with varying token capacities, letting users switch between quick answers or more in-depth explorations.

### 5. OpenRouter Integration & Model Customization

- **Default Models**\

  - **Extraction Model**: `google/gemini-flash-1.5` – optimized for high-frequency entity extraction.\

  - **Answer Model**: `microsoft/phi-4` – a reasoning-focused model trained on top-tier datasets.

- **Flexible Overrides**\
  Users can swap in other models for extraction or answer generation, fine-tuning performance for specialized tasks.

- **Temperature Control**\
  GPT temperature parameter (0 to 1) lets users tailor output creativity vs. determinism.

### 6. Local Model Integration & Parallel Processing

- **Cost Reduction**\
  Embedding, reranking, and translation models (e.g., `thenlper/gte-small`, `cross-encoder/ms-marco-MiniLM-L-12-v2`) run locally, cutting down on recurring API costs.

- **Concurrent Workloads**\
  Uses multi-core parallelism to handle large-scale document processing efficiently, although actual speed depends on system resources.

### 7. Bilingual Functionality

- **Auto Detection**\
  The app detects user language. By default, it generates English responses followed by Arabic translations.

- **Translation Control**\
  Users can disable Arabic translation for quicker performance.

- **English Document Focus**\
  While bilingual, the system is optimized for English-language documents.

---

## Usage and User Guidance

### 0. Obtain an OpenRouter API Key (If Required)

Visit [https://openrouter.ai](https://openrouter.ai) to register and obtain an OpenRouter API key. And fill it in the dedicated box in the setting section in when you start the application for the 1st time:

```env
OPENROUTER_API_KEY=sd-.....
```

### 1. Clone the Repository

```bash
git clone https://github.com/nlmatics/SocioPoliticsGraphMind.git
cd SocioPoliticsGraphMind
```

### 2. Create a Virtual Environment (Recommended)

Using **Conda**:

```bash
conda create -n socioenv python=3.12.8
conda activate socioenv
pip install --upgrade pip==25.0
```

Using **Python venv**:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip==25.0
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Then navigate to [http://localhost:5000](http://localhost:5000) in your browser to begin using **SocioPolitics GraphMind**.

Once running, you can:

- **Upload Documents**: Access the `/upload-files` endpoint or interface to upload your documents.
- **Process Documents**: Send a request to `/process-documents` to segment, embed, and build the knowledge graph.
- **Perform Searches**: Use `/search` with a text query. By default, results appear in English, followed by an Arabic translation if enabled.
