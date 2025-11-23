# Probabilistic Information Retrieval System

## Overview

This project implements probabilistic information retrieval systems using multiple ranking algorithms: the Binary Independence Model (BIM) and BM25 (Okapi Best Matching 25). The system processes a collection of documents and retrieves the most relevant documents for given search queries using advanced probabilistic and statistical methods.

## Features

- **Document Preprocessing**: Tokenizes and normalizes text documents for indexing
- **Query Processing**: Processes user queries and extracts relevant terms
- **Statistical Analysis**: Computes term frequencies, document frequencies, and collection statistics
- **Multiple Ranking Models**: 
  - Binary Independence Model (BIM) - probabilistic ranking approach
  - BM25 (Okapi) - state-of-the-art ranking algorithm with term frequency saturation
  - Language Model with Jelinek-Mercer (LM-JM) - smoothing-based ranking
- **Comprehensive Results**: Generates ranked results from all models and outputs them to files

## Project Structure

```
.
├── README.md                                    # Project documentation
├── Shreejal KC_W4_Week 4 Assignment.ipynb      # BIM implementation notebook
├── Probabilistic_IR_BM25.ipynb                  # BM25 and LM-JM implementation notebook
├── documents/
│   ├── speeches/                                # Folder containing document collection
│   │   └── speech_*.txt                         # Individual text documents
│   └── query/
│       └── queries1.txt                         # File containing search queries
├── results.txt                                  # BIM output results
└── resultsBM25.txt                              # BM25 and LM-JM output results
```

## How It Works

### 1. **Preprocessing**
- Converts text to lowercase
- Extracts words using regex pattern matching

### 2. **Document Loading**
- Reads all `.txt` files from the Trump Speeches folder
- Preprocesses each document

### 3. **Query Loading**
- Reads queries from `queries1.txt` file
- One query per line

### 4. **Statistical Computation**
- Calculates term frequency (TF) for each document
- Calculates document frequency (DF) for each term
- Stores document count

## Ranking Algorithms

### Binary Independence Model (BIM)
The Binary Independence Model calculates relevance using probabilistic estimation:
$$\text{Score} = \prod_{term \in query} \frac{P(term|relevant)}{P(term|not\_relevant)}$$

**Characteristics:**
- Probabilistic framework based on binary term occurrence
- Assumes term independence
- Uses Laplace smoothing for probability estimation

### Okapi BM25
BM25 is a modern ranking function that improves upon TF-IDF by incorporating:
$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

**Advantages of BM25:**
- **Term Frequency Saturation**: Prevents very high term frequencies from dominating scores
- **Document Length Normalization**: Accounts for document length to avoid bias toward longer documents
- **Tunable Parameters**: k₁ (default 1.2) and b (default 0.75) allow fine-tuning for specific domains
- **Industry Standard**: Widely used in production search engines and information retrieval systems
- **Empirically Proven**: Consistently outperforms simpler models like TF-IDF
- **Robust**: Handles edge cases like very common or very rare terms effectively

**Parameters:**
- k₁ = 1.2 (Term frequency saturation parameter)
- b = 0.75 (Document length normalization parameter)

### Language Model with Jelinek-Mercer (LM-JM) Smoothing
A probabilistic model that combines document and collection probabilities:

**Characteristics:**
- Smooths document language model with collection language model
- Uses log probability to avoid numerical underflow
- λ parameter (default 0.7) controls the balance between document and collection models
- Effective for handling sparse term occurrences

## Comparison of Approaches

| Aspect | BIM | BM25 | LM-JM |
|--------|-----|------|-------|
| Framework | Probabilistic | Empirical + Statistical | Probabilistic |
| TF Saturation | No | Yes | Via smoothing |
| Doc Length Norm | No | Yes | Yes |
| Parameters | Fixed | Tunable | Tunable (λ) |
| Complexity | Low | Medium | Medium |
| Performance | Good | Excellent | Very Good |

## Installation

### Requirements
- Python 3.6+
- Libraries: `os`, `re`, `numpy`, `collections`, `math`

### Setup
1. Clone the repository
2. Prepare your document collection in the `Trump Speeches/` folder
3. Create a `queries1.txt` file with one query per line

## Usage

### Running the System

```python
# Simply execute the Jupyter notebook, or run the main function:
if __name__ == "__main__":
    main()
```

### Input Format

**queries1.txt** - One query per line:
```
healthcare policy
economic growth
foreign relations
```

### Output Format

**results.txt** - Ranked documents with scores:
```
Query: healthcare policy
Document: speech1.txt, Score: 0.8524
Document: speech3.txt, Score: 0.7231
Document: speech2.txt, Score: 0.5120

Query: economic growth
Document: speech2.txt, Score: 0.9123
...
```

## Key Functions

| Function | Purpose | Model |
|----------|---------|-------|
| `preprocess(text)` | Tokenizes and normalizes text | Both |
| `load_documents(folder_path)` | Loads documents from folder | Both |
| `load_queries(query_file_path)` | Loads queries from file | Both |
| `compute_statistics(docs)` | Computes TF, DF, and collection statistics | Both |
| `compute_relevance_prob(query, ...)` | Calculates BIM scores | BIM |
| `compute_bm25_score(query, ...)` | Calculates BM25 scores | BM25 |
| `compute_lm_jm_score(query, ...)` | Calculates LM-JM smoothed scores | LM-JM |
| `retrieve_documents(folder_path, query_file_path)` | BIM retrieval pipeline | BIM |
| `retrieve_documents_and_result(folder_path, query_file_path, output_file)` | BM25 and LM-JM retrieval pipeline | BM25/LM-JM |
| `main()` | Entry point for execution | Both |

## Output

Results are saved to separate files for each model:

**results.txt** - BIM model output format:
```
Query: healthcare policy
Document: speech_1.txt, Score: 0.8524
Document: speech_3.txt, Score: 0.7231
Document: speech_2.txt, Score: 0.5120
========================================

Query: economic growth
Document: speech_2.txt, Score: 0.9123
...
```

**resultsBM25.txt** - BM25 and LM-JM model output format:
```
Query: healthcare policy
----------------------------------------
Results (Okapi BM25):
  Document: speech_1.txt, Score: 0.9234
  Document: speech_3.txt, Score: 0.8567
  Document: speech_2.txt, Score: 0.7891

Results (LM with Jelinek-Mercer):
  Document: speech_1.txt, Score: -5.4321
  Document: speech_3.txt, Score: -5.6543
  Document: speech_2.txt, Score: -6.1234
========================================
```

## Notes

- **BIM Implementation**: Uses Laplace smoothing for probability estimation; documents ranked by probability ratio
- **BM25 Implementation**: Industry-standard ranking with term saturation and document length normalization
- **LM-JM Implementation**: Language model smoothing with collection probabilities to handle sparse data
- **Document Length Normalization**: BM25 and LM-JM account for varying document lengths to prevent bias
- **All Results**: Every document is ranked and output for each query

## Performance Characteristics

- **BM25**: Generally provides the best ranking quality, especially for large collections
- **LM-JM**: Excellent for handling vocabulary coverage and zero-probability problems
- **BIM**: Good baseline with probabilistic foundations; simpler but less adaptive than BM25

## Future Enhancements

- Implement Dirichlet Prior smoothing for Language Models
- Add support for stemming and lemmatization
- Implement query expansion and pseudo-relevance feedback
- Add visualization of ranking differences across models
- Create web interface for retrieval system
- Implement indexing for faster retrieval on large collections
- Add evaluation metrics (NDCG, MAP, MRR)

## License

This project is provided for educational purposes.
