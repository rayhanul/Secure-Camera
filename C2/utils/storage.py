# Untrusted Mongo Database - hashed(key), hashed(value) store

# Key -> ENC(Embedding)
# Val -> Data = [ENC(Object Data 1), ENC(Object Data 2), ... ]

# If embedding is exactly same, we can encrypt the same image and put it in the value data array.


import base64
import json
import os
import pickle
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import List

import numpy as np
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pymongo import MongoClient
from utils.reid_result import ReIDResult

MASTER_ENC_KEY = base64.b64decode(os.environ["ENC_KEY_BASE64"])


class SecureReIDStorage:
    def __init__(self):
        mongo_host_url = os.environ["MONGO_HOST_URL"]
        mongo_user = os.environ["MONGO_ROOT_USERNAME"].strip()
        mongo_pass = os.environ["MONGO_ROOT_PASSWORD"].strip()

        print(f" user name: {mongo_user} and pass: {mongo_pass}" )
        mongo_db_url = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host_url}"
        # mongo_db_url = os.getenv("MONGO_HOST_URL", "mongodb://localhost:27017")
        print(f"Connecting to MongoDB at {mongo_db_url}...")
        client = MongoClient(mongo_db_url)
        db = client[os.environ["MONGO_DB_NAME"]]
        self.aesgcm = AESGCM(MASTER_ENC_KEY)
        self.collection = db[os.environ["MONGO_COLLECTION_NAME"]]

    def _hash_embedding(self, embedding_vector: np.ndarray) -> str:
        # creating hash of embedding for MongoDB key

        # Input validation
        if not isinstance(embedding_vector, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(embedding_vector)}")

        if embedding_vector.size == 0:
            raise ValueError("Embedding vector is empty")

        if embedding_vector.dtype != np.float32:
            embedding_vector = embedding_vector.astype(np.float32)

        # has the raw bytes
        embedding_bytes = embedding_vector.tobytes()
        return sha256(embedding_bytes).hexdigest()

    # Enc
    def _encrypt_data(self, data: bytes) -> bytes:
        import os

        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    # Dec using AES-GCM
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        nonce = encrypted_data[:12]  # extract the nonce
        ciphertext = encrypted_data[12:]  # cipher text extract
        return self.aesgcm.decrypt(nonce, ciphertext, None)

    # Convert ReIDResult to bytes for encryption
    def _serialize_result(self, result: ReIDResult) -> bytes:
        # using pickle library for serialization and decerialization
        return pickle.dumps(result)

    # After Dec, decerailization
    def _deserialize_result(self, data: bytes) -> ReIDResult:
        return pickle.loads(data)

    def get_similar_results(self, embedding_vector: np.ndarray) -> List[ReIDResult]:
        """Retrieve all results for the same embedding"""
        # Create hash key for the embedding
        embedding_hash = self._hash_embedding(embedding_vector)
        return self.get_similar_results_by_hash(embedding_hash)

    def get_similar_results_by_hash(self, embedding_hash: str) -> List[ReIDResult]:
        """Retrieve all results for a specific embedding hash"""
        # Find document with this embedding
        doc = self.collection.find_one({"_id": embedding_hash})

        if not doc:
            print(f"No results found for embedding {embedding_hash[:8]}...")
            return []

        # Decrypt and deserialize all results
        results = []
        for encrypted_result in doc["results"]:
            try:
                decrypted_bytes = self._decrypt_data(encrypted_result)
                result = self._deserialize_result(decrypted_bytes)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not decrypt result: {e}")

        print(f"Retrieved {len(results)} results for embedding {embedding_hash[:8]}...")
        return results

    # store to Mongo
    def store_reid_result(self, embedding_vector: np.ndarray, result: ReIDResult):
        embedding_hash = self._hash_embedding(embedding_vector)
        # serialize and enc
        result_bytes = self._serialize_result(result)
        encrypted_result = self._encrypt_data(result_bytes)

        # check if embedding alreayd exists
        existing_doc = self.collection.find_one({"_id": embedding_hash})

        if existing_doc:
            self.collection.update_one(
                {"_id": embedding_hash}, {"$push": {"results": encrypted_result}}
            )
            print(f"Added result to existing embedding {embedding_hash[:8]}...")

        else:
            # new doc
            self.collection.insert_one(
                {"_id": embedding_hash, "results": [encrypted_result]}
            )
            print(f"Created new embedding entry {embedding_hash[:8]}")

    # retrieve all results of same embedding
    def get_results(self, embedding_vector: np.ndarray) -> List[ReIDResult]:
        embedding_hash = self._hash_embedding(embedding_vector)

        doc = self.collection.find_one({"_id": embedding_hash})

        if not doc:
            print(f"No results found for embedding {embedding_hash[:8]}...")
            return []

        # dec and deserialize tthe results
        results = []
        for encrypted_result in doc["results"]:
            try:
                decrypted_bytes = self._decrypt_data(encrypted_result)
                result = self._deserialize_result(decrypted_bytes)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not decrypt result: {e}")

        print(f"Retrieved {len(results)} results for embedding {embedding_hash[:8]}...")
        return results

    def close(self):
        self.client.close()
      

