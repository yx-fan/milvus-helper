import pytest
import numpy as np
from milvus_helper import MilvusClient
from pymilvus import list_collections, Collection

@pytest.fixture(scope="module")
def client():
    c = MilvusClient()
    yield c
    # Cleanup all test collections after all tests
    for name in list_collections():
        if name.startswith("test_"):
            Collection(name).drop()

@pytest.fixture(scope="module")
def setup_collection(client):
    name = "test_client_collection"
    client.delete_collection(name)
    client.create_collection(name, dim=64)
    vectors = np.random.rand(3, 64).tolist()
    metadata = [{"source": "test"} for _ in range(3)]
    client.insert_vectors(name, vectors, metadata)
    client.create_index(name)
    return name

def test_create_collection(client):
    name = "test_create"
    client.delete_collection(name)
    client.create_collection(name, dim=32)
    assert name in client.list_collections()

def test_create_collection_duplicate(client):
    name = "test_create"
    client.create_collection(name, dim=32)  # duplicate creation should pass silently

def test_create_collection_invalid_dim(client):
    with pytest.raises(Exception):
        client.create_collection("invalid", dim=-1)

def test_insert_vectors_success(client, setup_collection):
    vectors = np.random.rand(2, 64).tolist()
    metadata = [{"source": "unit"} for _ in range(2)]
    result = client.insert_vectors(setup_collection, vectors, metadata)
    assert isinstance(result, list)
    assert len(result) == 2

def test_insert_empty_vectors(client, setup_collection):
    result = client.insert_vectors(setup_collection, [], [])
    assert result is None

def test_insert_vectors_mismatch(client, setup_collection):
    vectors = np.random.rand(2, 64).tolist()
    metadata = [{"only": "one"}]
    result = client.insert_vectors(setup_collection, vectors, metadata)
    assert result is None

def test_search_vectors_success(client, setup_collection):
    vec = np.random.rand(64).tolist()
    results = client.search_vectors(setup_collection, vec, top_k=2, output_fields=["source"])
    assert isinstance(results, list)
    if results:
        assert "source" in results[0]
        assert "distance" in results[0]

def test_search_wrong_dim(client, setup_collection):
    vec = np.random.rand(10).tolist()
    results = client.search_vectors(setup_collection, vec)
    assert results == []

def test_search_nonexistent_collection(client):
    vec = np.random.rand(64).tolist()
    results = client.search_vectors("no_collection", vec)
    assert results == []

def test_get_collection_schema(client, setup_collection):
    schema = client.get_collection_schema(setup_collection)
    assert isinstance(schema, dict)
    assert "fields" in schema

def test_get_collection_schema_nonexistent(client):
    assert client.get_collection_schema("fake_collection") is None

def test_count_entities(client, setup_collection):
    count = client.count_entities(setup_collection)
    assert isinstance(count, int)
    assert count >= 3

def test_delete_collection(client):
    name = "test_delete"
    client.create_collection(name, dim=64)
    result = client.delete_collection(name)
    assert result
    assert name not in client.list_collections()

def test_delete_collection_nonexistent(client):
    result = client.delete_collection("no_such_collection")
    assert result is False

def test_get_collection(client, setup_collection):
    collection = client.get_collection(setup_collection)
    assert isinstance(collection, Collection)

def test_get_collection_nonexistent(client):
    assert client.get_collection("ghost") is None

def test_list_collections(client, setup_collection):
    collections = client.list_collections()
    assert isinstance(collections, list)
    assert setup_collection in collections
