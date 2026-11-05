# Document Question-Answering Assistant

A local AI-powered Document Question-Answering Assistant that allows users to ask questions and receive answers based on provided text documents.

## Features
- Loads and processes multiple `.txt` documents.
- Splits documents into smaller text chunks.
- Uses Sentence Transformers for semantic embeddings.
- Retrieves relevant passages using cosine similarity.
- Uses an extractive QA model to find answers.
- Displays source documents, chunk IDs, and supporting passages.
- Rejects unsupported questions with: `Information not found in provided document`.
- Provides an interactive interface using ipywidgets.

## Technologies
- Python
- Sentence Transformers
- Hugging Face Transformers
- Scikit-learn
- NumPy
- ipywidgets
- Jupyter Notebook

## Sample Documents
- `machine_learning.txt`
- `python_basics.txt`
- `university_rules.txt`

## How to Run
1. Clone this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Run the cells in order.
5. Ask questions through the interactive interface.

**Note:** The embedding and QA models run locally after their initial download. Similarity and confidence thresholds are configurable and may require adjustment for different documents.

## Project Goal
To explore semantic search, information retrieval, extractive question answering, and evidence-based responses using open-source tools.