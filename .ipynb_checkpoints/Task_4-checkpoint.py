#!/usr/bin/env python
# coding: utf-8

# # Task 4: Document Question-Answering Assistant
# 
# ## Objective
# 
# Build a local document question-answering system that retrieves relevant information from provided documents and generates answers supported by source passages.
# 
# ## Project Goals
# 
# - Load and process text documents.
# - Divide documents into meaningful chunks.
# - Retrieve relevant passages using local semantic embeddings.
# - Generate answers using extracted source evidence.
# - Display supporting passages with every answer.
# - Reject unsupported questions transparently.
# 
# ## Technology Stack
# 
# Python, Pandas, NLTK, Sentence-Transformers, Scikit-learn, and Streamlit.

# In[1]:


import os

folders = [
    "data/documents",
    "models",
    "src"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully.")


# In[2]:


documents = {
    "python_basics.txt": """
Python Programming Basics

Python is a high-level, interpreted programming language.
It was created by Guido van Rossum and first released in 1991.

Python supports multiple programming paradigms, including
object-oriented, procedural, and functional programming.

Variables are used to store data values.
Python has built-in data types such as integers, floats,
strings, lists, tuples, dictionaries, and sets.

Functions are reusable blocks of code defined using the
def keyword. They help organize programs and reduce repetition.
""",

    "machine_learning.txt": """
Introduction to Machine Learning

Machine Learning is a branch of artificial intelligence
that enables computers to learn patterns from data.

Supervised learning uses labeled training data.
Its common tasks include classification and regression.

Unsupervised learning works with unlabeled data.
Clustering is one of its common techniques.

A dataset is commonly divided into training and testing sets.
The training set is used to teach the model, while the
testing set evaluates its performance.

Common classification evaluation metrics include accuracy,
precision, recall, F1-score, and ROC-AUC.
""",

    "university_rules.txt": """
University Academic Rules

Students must maintain regular attendance throughout
the academic semester.

The minimum attendance requirement is 75 percent
to become eligible for final examinations.

Assignments must be submitted before their deadlines.
Late submissions may result in penalties.

Students must carry their university identification cards
while entering the campus.

The final examination schedule is announced by the
university administration.
"""
}

for filename, content in documents.items():
    path = os.path.join("data/documents", filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content.strip())

print("Sample documents created:")
print(os.listdir("data/documents"))


# In[3]:


def load_documents(folder_path):
    loaded_documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            loaded_documents.append({
                "filename": filename,
                "content": content
            })

    return loaded_documents


loaded_documents = load_documents("data/documents")

print("Total documents loaded:", len(loaded_documents))

for document in loaded_documents:
    print("\nFilename:", document["filename"])
    print("Characters:", len(document["content"]))
    print("Preview:", document["content"][:100])


# In[4]:


import re

def chunk_documents(documents, chunk_size=3):
    chunks = []

    for document in documents:
        sentences = re.split(
            r'(?<=[.!?])\s+',
            document["content"].strip()
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(
                sentences[i:i + chunk_size]
            )

            chunks.append({
                "source": document["filename"],
                "chunk_id": len(chunks),
                "text": chunk_text
            })

    return chunks


chunks = chunk_documents(loaded_documents, chunk_size=3)

print("Total chunks created:", len(chunks))

for chunk in chunks:
    print(f"\nChunk ID: {chunk['chunk_id']}")
    print("Source:", chunk["source"])
    print("Text:", chunk["text"])


# In[5]:


get_ipython().run_line_magic('pip', 'install -q sentence-transformers')


# In[6]:


from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# In[7]:


chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embeddings shape:", chunk_embeddings.shape)
print("Total embedded chunks:", len(chunk_embeddings))


# In[8]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_passages(question, top_k=3):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results


print("Retrieval function created successfully.")


# In[9]:


question = "Who created Python and when was it released?"

results = retrieve_passages(question)

for result in results:
    print("Source:", result["source"])
    print("Similarity:", round(result["score"], 4))
    print("Passage:", result["text"])
    print("-" * 60)


# In[10]:


question = "What is the capital of France?"

results = retrieve_passages(question)

for result in results:
    print("Source:", result["source"])
    print("Similarity:", round(result["score"], 4))
    print("Passage:", result["text"])
    print("-" * 60)


# In[11]:


SIMILARITY_THRESHOLD = 0.30

def answer_question(question):
    results = retrieve_passages(question, top_k=3)

    best_result = results[0]

    if best_result["score"] < SIMILARITY_THRESHOLD:
        print("Information not found in provided document")
        return

    print("Answer supported by document:")
    print(best_result["text"])

    print("\nSource:", best_result["source"])
    print("Chunk ID:", best_result["chunk_id"])
    print("Similarity:", round(best_result["score"], 4))


print("Question-answering function created successfully.")


# In[12]:


print("TEST 1: Supported question")
answer_question("Who created Python and when was it released?")

print("\n" + "=" * 60)

print("TEST 2: Unsupported question")
answer_question("What is the capital of France?")


# In[13]:


test_questions = [
    "What is the minimum attendance required for final exams?",
    "How does supervised learning work?",
    "Which data types are available in Python?",
    "What happens if students submit assignments late?",
    "How much is the university tuition fee?"
]

for question in test_questions:
    print("\nQuestion:", question)
    answer_question(question)
    print("-" * 70)


# In[14]:


SIMILARITY_THRESHOLD = 0.40

for question in test_questions:
    print("\nQuestion:", question)
    answer_question(question)
    print("-" * 70)


# In[15]:


paraphrased_questions = [
    "Who is the person behind the creation of Python?",
    "What percentage of attendance must students maintain?",
    "What is the purpose of a training dataset?",
    "Name some common Python collection and data types.",
    "Can students take final exams with low attendance?"
]

for question in paraphrased_questions:
    print("\nQuestion:", question)
    answer_question(question)
    print("-" * 70)


# In[16]:


def extract_best_sentence(question, passage):
    sentences = re.split(r'(?<=[.!?])\s+', passage.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(scores)

    return sentences[best_index]


print("Sentence extraction function created successfully.")


# In[17]:


question = "What is the purpose of a training dataset?"

results = retrieve_passages(question)

best_passage = results[0]["text"]

answer = extract_best_sentence(question, best_passage)

print("Question:", question)
print("\nExtracted answer:", answer)
print("\nOriginal supporting passage:", best_passage)
print("\nSource:", results[0]["source"])


# In[18]:


def answer_question(question):
    results = retrieve_passages(question, top_k=3)

    best_result = results[0]

    if best_result["score"] < SIMILARITY_THRESHOLD:
        print("Information not found in provided document")
        return

    extracted_answer = extract_best_sentence(
        question,
        best_result["text"]
    )

    print("Answer:", extracted_answer)

    print("\nSupporting passage:")
    print(best_result["text"])

    print("\nSource:", best_result["source"])
    print("Chunk ID:", best_result["chunk_id"])
    print("Similarity:", round(best_result["score"], 4))


print("Updated question-answering function created successfully.")


# In[19]:


final_test_questions = [
    "Who created Python?",
    "What percentage of attendance is required for exams?",
    "What is the capital of France?"
]

for question in final_test_questions:
    print("\nQuestion:", question)
    answer_question(question)
    print("\n" + "=" * 70)


# In[20]:


def extract_best_sentence(question, passage):
    sentences = re.split(r'(?<=[.!?])\s+', passage.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Remove short document headings
    sentences = [
        sentence for sentence in sentences
        if len(sentence.split()) > 4
    ]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(scores)

    return sentences[best_index]


print("Improved sentence extraction function created.")


# In[21]:


question = "Who created Python?"
results = retrieve_passages(question)

answer = extract_best_sentence(
    question,
    results[0]["text"]
)

print("Extracted answer:", answer)


# In[22]:


def extract_best_sentence(question, passage):
    clean_passage = " ".join(passage.split())

    sentences = re.split(r'(?<=[.!?])\s+', clean_passage)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.split()) > 4
        and not sentence.strip().endswith("Basics")
    ]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(scores)

    return sentences[best_index]


print("Sentence extraction function updated.")


# In[23]:


question = "Who created Python?"
results = retrieve_passages(question)

print(extract_best_sentence(question, results[0]["text"]))


# In[24]:


def extract_best_sentence(question, passage):
    clean_passage = " ".join(passage.split())

    sentences = re.split(r'(?<=[.!?])\s+', clean_passage)

    # Remove document headings
    sentences = [
        sentence for sentence in sentences
        if len(sentence.split()) > 4
        and sentence not in [
            "Python Programming Basics",
            "Introduction to Machine Learning",
            "University Academic Rules"
        ]
    ]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(scores)

    return sentences[best_index]


print("Sentence extraction function updated.")


# In[25]:


question = "Who created Python?"
results = retrieve_passages(question)

print(extract_best_sentence(question, results[0]["text"]))


# In[26]:


question = "Who created Python?"
results = retrieve_passages(question)

passage = " ".join(results[0]["text"].split())

sentences = re.split(r'(?<=[.!?])\s+', passage)

for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")


# In[27]:


def extract_best_sentence(question, passage):
    clean_passage = " ".join(passage.split())

    headings = [
        "Python Programming Basics",
        "Introduction to Machine Learning",
        "University Academic Rules"
    ]

    for heading in headings:
        clean_passage = clean_passage.replace(heading, "")

    sentences = re.split(r'(?<=[.!?])\s+', clean_passage.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(scores)

    return sentences[best_index]


print("Sentence extraction function fixed.")


# In[28]:


question = "Who created Python?"
results = retrieve_passages(question)

print(extract_best_sentence(question, results[0]["text"]))


# In[29]:


question = "Who created Python?"
results = retrieve_passages(question)

passage = " ".join(results[0]["text"].split())

for heading in [
    "Python Programming Basics",
    "Introduction to Machine Learning",
    "University Academic Rules"
]:
    passage = passage.replace(heading, "")

sentences = re.split(r'(?<=[.!?])\s+', passage.strip())

question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)

sentence_embeddings = embedding_model.encode(
    sentences,
    normalize_embeddings=True
)

scores = cosine_similarity(
    question_embedding,
    sentence_embeddings
)[0]

for sentence, score in zip(sentences, scores):
    print(round(float(score), 4), ":", sentence)


# In[30]:


def extract_best_sentence(question, passage):
    clean_passage = " ".join(passage.split())

    headings = [
        "Python Programming Basics",
        "Introduction to Machine Learning",
        "University Academic Rules"
    ]

    for heading in headings:
        clean_passage = clean_passage.replace(heading, "")

    sentences = re.split(r'(?<=[.!?])\s+', clean_passage.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    question_lower = question.lower()

    # Prioritize sentences containing names for "Who" questions
    if question_lower.startswith("who"):
        for sentence in sentences:
            if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', sentence):
                return sentence

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    return sentences[np.argmax(scores)]


print("Question-aware extraction function created.")


# In[31]:


question = "Who created Python?"
results = retrieve_passages(question)

print(extract_best_sentence(question, results[0]["text"]))


# In[33]:


def extract_best_sentence(question, passage):
    clean_passage = " ".join(passage.split())

    headings = [
        "Python Programming Basics",
        "Introduction to Machine Learning",
        "University Academic Rules"
    ]

    for heading in headings:
        clean_passage = clean_passage.replace(heading, "")

    sentences = re.split(r'(?<=[.!?])\s+', clean_passage.strip())

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    question_lower = question.lower()

    if question_lower.startswith("who"):
        for sentence in sentences:
            if re.search(
                r'\b(created|developed|founded|invented|written|designed)\s+by\b',
                sentence,
                re.IGNORECASE
            ):
                return sentence

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    sentence_embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        sentence_embeddings
    )[0]

    return sentences[np.argmax(scores)]


# In[34]:


question = "Who created Python?"
results = retrieve_passages(question)

print(extract_best_sentence(question, results[0]["text"]))


# In[35]:


test_questions = [
    "Who created Python?",
    "What is the minimum attendance requirement?",
    "What is the purpose of a testing dataset?",
    "What is the capital of France?"
]

for question in test_questions:
    print(f"\nQuestion: {question}")
    answer_question(question)


# In[36]:


test_questions = [
    "What is the tuition fee of the university?",
    "Who is the current president of the university?",
    "Which programming language is faster, Python or C++?",
    "What is the minimum attendance requirement?"
]

for question in test_questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    answer_question(question)


# In[37]:


question = "Which programming language is faster, Python or C++?"

results = retrieve_passages(question)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Passage: {result['text']}\n")


# In[38]:


comparison_words = [
    "faster", "slower", "better", "worse",
    "difference", "compare", "versus", "vs"
]

test_questions = [
    "Which programming language is faster, Python or C++?",
    "What is the difference between classification and regression?",
    "Who created Python?",
    "Is Python a high-level language?"
]

for question in test_questions:
    is_comparison = any(
        word in question.lower()
        for word in comparison_words
    )

    print(f"{question}")
    print(f"Comparison question: {is_comparison}\n")


# In[41]:


def answer_question(question):
    results = retrieve_passages(question)

    if not results or results[0]["score"] < SIMILARITY_THRESHOLD:
        print("Information not found in provided document")
        return

    comparison_words = [
        "faster", "slower", "better", "worse",
        "difference", "compare", "versus", "vs"
    ]

    question_lower = question.lower()

    is_comparison = any(
        word in question_lower
        for word in comparison_words
    )

    if is_comparison:
        comparison_evidence = [
            "faster", "slower", "better", "worse",
            "difference", "compared with", "compared to",
            "whereas", "while", "unlike"
        ]

        passage_lower = results[0]["text"].lower()

        has_comparison_evidence = any(
            phrase in passage_lower
            for phrase in comparison_evidence
        )

        if not has_comparison_evidence:
            print("Information not found in provided document")
            return

    best_passage = results[0]

    answer = extract_best_sentence(
        question,
        best_passage["text"]
    )

    print(f"Answer: {answer}")
    print("\nSupporting passage:")
    print(best_passage["text"])
    print(f"\nSource: {best_passage['source']}")
    print(f"Chunk ID: {best_passage['chunk_id']}")
    print(f"Similarity: {best_passage['score']:.4f}")


# In[42]:


test_questions = [
    "Which programming language is faster, Python or C++?",
    "What is the difference between classification and regression?",
    "Who created Python?"
]

for question in test_questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    answer_question(question)


# In[43]:


test_questions = [
    "What are the common classification evaluation metrics?",
    "What does supervised learning use?",
    "What is the minimum attendance needed for final exams?",
    "Which keyword is used to define functions in Python?"
]

for question in test_questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    answer_question(question)


# In[ ]:




