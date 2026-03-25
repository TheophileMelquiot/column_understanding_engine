# Column Understanding Engine

A modular machine learning system for **semantic understanding of tabular data** extracted from raw, unstructured Excel files.

## Problem Statement

Real-world Excel files are messy: they contain multiple tables per sheet, merged cells, hierarchical headers, and inconsistent formatting. This project builds an end-to-end pipeline that:

1. **Detects** and extracts multiple tables from a raw Excel file
2. **Reconstructs** complex headers (including merged and hierarchical headers)
3. **Cleans** and structures each table into a usable DataFrame
4. **Extracts features** (statistical, pattern-based, and textual) from each column
5. **Classifies** each column semantically using ML/DL models
6. **Evaluates** results with rigorous metrics, benchmarking, and ablation studies

## Architecture

```
Raw Excel File
      │
      ▼
┌─────────────────┐
│ Table Detection  │  BFS-based connected component detection
└────────┬────────┘
         ▼
┌─────────────────────┐
│ Header Reconstruction│  Merged cell propagation, hierarchical flattening
└────────┬────────────┘
         ▼
┌─────────────────┐
│  Data Cleaning   │  Noise removal, value normalization
└────────┬────────┘
         ▼
┌──────────────────────┐
│  Feature Engineering  │  Statistical + Pattern (regex) + Textual (BERT)
└────────┬─────────────┘
         ▼
┌─────────────────┐
│    Modeling      │  Logistic Regression │ XGBoost │ PyTorch MLP
└────────┬────────┘
         ▼
┌─────────────────┐
│   Evaluation     │  Accuracy, F1, ROC-AUC, ablation study
└─────────────────┘
```

## Repository Structure

```
column_engine/              # Main Python package
├── __init__.py
├── table_detection.py      # Step 1 — BFS connected component detection
├── header_reconstruction.py# Step 2 — Merged cell & hierarchical header handling
├── data_cleaning.py        # Step 3 — Noise removal, normalization
├── feature_engineering.py  # Step 4 — Statistical, pattern, textual features
├── evaluation.py           # Step 6 — Metrics, benchmarking, ablation
├── pipeline.py             # End-to-end pipeline
└── models/                 # Step 5 — Classification models
    ├── __init__.py
    ├── base.py             # Abstract base classifier
    ├── logistic.py         # Logistic Regression baseline
    ├── xgboost_model.py    # XGBoost strong baseline
    └── deep_model.py       # PyTorch MLP with embedding fusion

tests/                      # Unit tests
├── conftest.py             # Shared test fixtures (Excel generators)
├── test_table_detection.py
├── test_header_reconstruction.py
├── test_data_cleaning.py
├── test_feature_engineering.py
├── test_models.py
└── test_pipeline.py
```

## Methodology

### Table Detection

Treats each worksheet as a 2D grid. Non-empty cells are grouped into connected components via BFS. Small components (< `min_cells`) are filtered as noise.

### Header Reconstruction

Uses `openpyxl` merged-cell metadata to propagate parent header values across merged regions. Multi-level headers are flattened by joining with `"_"` (e.g., `Sales` + `Q1` → `Sales_Q1`).

### Feature Engineering

| Feature Group  | Features                                      |
|---------------|-----------------------------------------------|
| Statistical   | mean, std, n_unique, entropy, null_ratio      |
| Pattern       | is_email, is_phone, is_price, is_date, is_uuid|
| Textual       | Header embedding, sampled-values embedding (BERT/DistilBERT) |

### Models

| Model             | Type             | Description                              |
|-------------------|------------------|------------------------------------------|
| Logistic Regression | Baseline        | Standardized features + multinomial LR   |
| XGBoost           | Strong baseline  | Gradient-boosted trees                   |
| Deep MLP          | Deep Learning    | BERT embeddings + tabular features → MLP |

The deep model architecture:
- Linear projection of text embeddings
- Concatenation with engineered tabular features
- Two hidden layers with ReLU activation and dropout
- Softmax classification head

### Evaluation

- **Metrics**: Accuracy, macro F1-score, ROC-AUC (one-vs-rest)
- **Benchmark**: Side-by-side comparison of all three models
- **Ablation study**: Systematic removal of feature groups to measure contribution

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from column_engine.pipeline import process_excel

# Run the full pipeline on an Excel file
result = process_excel("data/my_file.xlsx")

# Inspect results
for sheet_name, tables in result.items():
    for table_key, table_info in tables.items():
        print(f"\n{sheet_name} / {table_key}")
        print(f"  Shape: {table_info['dataframe'].shape}")
        for col in table_info['columns']:
            print(f"  Column: {col['name']} → type={col['type']}")
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Expected Output

For each detected table, the pipeline produces:

```python
{
  "columns": [
    {
      "name": "Sales_Q1",
      "predicted_label": "revenue",  # when classifier is provided
      "type": "numeric",
      "confidence": 0.92
    }
  ]
}
```

## Technical Choices

- **openpyxl** for Excel parsing: provides access to merged-cell metadata and cell-level formatting
- **BFS** for table detection: simple, robust, and handles arbitrary table layouts
- **scikit-learn** for classical ML: well-tested, efficient implementations
- **PyTorch** for deep learning: flexible architecture design, GPU support
- **HuggingFace Transformers** for text embeddings: state-of-the-art pretrained models

## Limitations

- Table detection assumes tables are separated by empty rows/columns
- Header detection heuristic may fail on tables where headers contain numeric values
- Deep model performance depends on quality of text embeddings and training data size
- Current regex patterns cover common formats but may miss locale-specific variants

## License

MIT License — see [LICENSE](LICENSE) for details.
