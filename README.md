# Probabilistic Information Retrieval System

## Overview

This project implements a probabilistic information retrieval system using the Binary Independence Model (BIM) to rank documents based on relevance to user queries. The system processes a collection of documents and retrieves the most relevant documents for given search queries.

## Features

- **Document Preprocessing**: Tokenizes and normalizes text documents for indexing
- **Query Processing**: Processes user queries and extracts relevant terms
- **Statistical Analysis**: Computes term frequencies and document frequencies
- **Probabilistic Ranking**: Uses BIM (Binary Independence Model) to calculate relevance scores
- **Result Output**: Generates ranked results and outputs them to a file


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

### 5. **Relevance Calculation (BIM)**
The Binary Independence Model calculates relevance using:
$$\text{Score} = \prod_{term \in query} \frac{P(term|relevant)}{P(term|not\_relevant)}$$

### 6. **Result Generation**
- Ranks documents by relevance score in descending order
- Outputs results to `results.txt`

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

| Function | Purpose |
|----------|---------|
| `preprocess(text)` | Tokenizes and normalizes text |
| `load_documents(folder_path)` | Loads documents from folder |
| `load_queries(query_file_path)` | Loads queries from file |
| `compute_statistics(docs)` | Computes TF and DF statistics |
| `compute_relevance_prob(query, ...)` | Calculates BIM scores |
| `retrieve_documents(folder_path, query_file_path)` | Main retrieval pipeline |
| `main()` | Entry point for execution |

## Output

Results are saved to `results.txt` with the following format:
- Query line showing the user's search query
- Multiple document lines with filename and relevance score
- Blank line separator between queries

## Notes

- The BIM implementation uses Laplace smoothing for probability estimation
- Documents are ranked by relevance score in descending order
- Scores represent the probability ratio of document relevance

## Future Enhancements

- Implement BM25 (Best Matching 25) algorithm
- Add support for stemming and lemmatization
- Implement term weighting (TF-IDF)
- Add query expansion techniques
- Create visualization of results

## License

This project is provided for educational purposes.
