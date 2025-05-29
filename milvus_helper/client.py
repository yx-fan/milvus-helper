import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, list_collections, utility
)

class MilvusClient:
    def __init__(self, host: str = "localhost", port: str = "19530", alias: str = "default"):
        try:
            connections.connect(alias=alias, host=host, port=port)
            self.alias = alias
            logging.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise

    def get_collection(self, name: str) -> Optional[Collection]:
        if name not in list_collections(using=self.alias):
            logging.warning(f"Collection '{name}' does not exist.")
            return None
        return Collection(name, using=self.alias)

    def create_collection(self, name: str, dim: int = 64) -> None:
        if name in list_collections(using=self.alias):
            logging.info(f"Collection '{name}' already exists.")
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description=f"Collection '{name}' with dim={dim}",
            enable_dynamic_field=True
        )
        Collection(name=name, schema=schema, using=self.alias)
        logging.info(f"Created collection '{name}' with dimension {dim}.")

    def create_index(self, name: str, index_type: str = "IVF_FLAT", nlist: int = 128) -> bool:
        collection = self.get_collection(name)
        if not collection:
            return False

        if collection.indexes:
            logging.info(f"Collection '{name}' already has an index.")
            return False

        try:
            index_params = {
                "index_type": index_type,
                "metric_type": "L2",
                "params": {"nlist": nlist},
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logging.info(f"Created index for collection '{name}'.")
            return True
        except Exception as e:
            logging.error(f"Failed to create index: {e}")
            return False

    def insert_vectors(
        self,
        name: str,
        vectors: List[List[float]],
        extra_fields: Optional[List[Dict[str, Any]]] = None,
        auto_flush: bool = True,
        auto_load: bool = True
    ) -> Optional[List[int]]:
        collection = self.get_collection(name)
        if not collection:
            return None

        if extra_fields is None:
            extra_fields = [{} for _ in vectors]

        if len(vectors) != len(extra_fields):
            logging.error("Mismatch between number of vectors and extra_fields.")
            return None

        insert_data = [{"vector": v, **meta} for v, meta in zip(vectors, extra_fields)]

        try:
            res = collection.insert(insert_data)
            if auto_flush:
                collection.flush()
            if auto_load:
                collection.load()
            return list(res.primary_keys)
        except Exception as e:
            logging.error(f"Insert failed for collection '{name}': {e}")
            return None

    def search_vectors(
        self,
        name: str,
        query_vector: List[float],
        top_k: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        collection = self.get_collection(name)
        if not collection:
            return []

        try:
            collection.load()
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=output_fields,
                expr=filter_expr
            )

            hits = []
            for result in results[0]:
                record = {"id": result.id, "distance": result.distance}
                if output_fields:
                    record.update(result.entity.get(field, None) for field in output_fields)
                hits.append(record)
            return hits
        except Exception as e:
            logging.error(f"Search failed in '{name}': {e}")
            return []

    def count_entities(self, name: str) -> Optional[int]:
        collection = self.get_collection(name)
        if not collection:
            return None

        try:
            collection.load()
            return collection.num_entities
        except Exception as e:
            logging.error(f"Count failed for '{name}': {e}")
            return None

    def delete_collection(self, name: str) -> bool:
        if name not in list_collections(using=self.alias):
            logging.warning(f"Collection '{name}' does not exist.")
            return False

        try:
            Collection(name=name, using=self.alias).drop()
            logging.info(f"Deleted collection '{name}'.")
            return True
        except Exception as e:
            logging.error(f"Failed to delete collection '{name}': {e}")
            return False

    def list_collections(self) -> List[str]:
        try:
            return list_collections(using=self.alias)
        except Exception as e:
            logging.error(f"Failed to list collections: {e}")
            return []
    
    def get_collection_schema(self, name: str) -> Optional[dict]:
        try:
            collection = Collection(name)
            return {
                "fields": {
                    f.name: {
                        "type": f.dtype.name,
                        "is_primary": f.is_primary,
                        "auto_id": getattr(f, "auto_id", None)
                    }
                    for f in collection.schema.fields
                },
                "description": collection.schema.description
            }
        except Exception as e:
            logging.error(f"Failed to get schema for '{name}': {e}")
            return None