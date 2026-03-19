# pmetal dataset

Dataset utilities — analyze, download, and convert training data.

Utilities for working with training datasets.

## Subcommands

### analyze

Analyze a local dataset — shows format, row count, token statistics, and column info.

```bash
pmetal dataset analyze train.jsonl
pmetal dataset analyze data.parquet
```

### download

Download a dataset from HuggingFace Hub.

```bash
pmetal dataset download squad --output ./data/
```

### convert

Convert between dataset formats.

```bash
# Parquet to JSONL
pmetal dataset convert data.parquet --format jsonl --output data.jsonl

# ShareGPT to Alpaca
pmetal dataset convert sharegpt.json --format alpaca --output alpaca.jsonl
```

## Supported Formats

| Format | Extensions | Read | Write |
|--------|-----------|------|-------|
| JSONL | `.jsonl` | Yes | Yes |
| JSON | `.json` | Yes | Yes |
| Parquet | `.parquet` | Yes | Yes |
| CSV | `.csv` | Yes | Yes |

## See Also

- [pmetal train](/cli/train/) — Use datasets for training
