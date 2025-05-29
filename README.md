# Milvus Helper

A lightweight Python wrapper around [Milvus](https://milvus.io/) to simplify common vector database operations with `pymilvus`.

## Features

- Simplified collection management
- Easy vector insertion with optional metadata
- Basic search interface
- Built-in logging and error handling

## Installation

```bash
pip install git+https://github.com/yx-fan/milvus-helper.git
```

## Usage

```python
from milvus_helper import MilvusClient

client = MilvusClient()
client.create_collection("test_collection", dim=384)
client.insert_vectors("test_collection", [[0.1]*384])
results = client.search("test_collection", [0.1]*384)
print(results)
```

## Requirements

- Python 3.8+
- Milvus 2.x running locally or remotely
- `pymilvus` >= 2.4.3

## License

MIT License © 2025 [Yuxin Fan](mailto:lawrence.yuxinfan@outlook.com)

[![PyPI version](https://badge.fury.io/py/milvus-helper.svg)](https://pypi.org/project/milvus-helper/)
