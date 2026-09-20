import os
import json
from pathlib import Path
from typing import Dict, List

import psycopg
from psycopg.rows import dict_row


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dog_embedding_metadata (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    model TEXT NOT NULL,
    preprocessing TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0)
);

CREATE TABLE IF NOT EXISTS dog_embeddings (
    dog_id TEXT PRIMARY KEY,
    image_name TEXT NOT NULL,
    image_hash TEXT NOT NULL UNIQUE CHECK (char_length(image_hash) = 64),
    embedding DOUBLE PRECISION[] NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connection_string() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError(
            "DATABASE_URL is not set. Example: "
            "postgresql://postgres:postgres@localhost:5432/dogdb"
        )
    return value


def initialize_database() -> None:
    with psycopg.connect(connection_string()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)


def migrate_json_database(json_path: Path) -> bool:
    if not json_path.exists():
        return False

    with json_path.open("r", encoding="utf-8") as file:
        database = json.load(file)
    if not isinstance(database, dict) or not database.get("dogs"):
        return False
    if load_database().get("dogs"):
        return False

    save_database(database)
    return True


def load_database() -> Dict:
    with psycopg.connect(connection_string(), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT model, preprocessing, embedding_dimension "
                "FROM dog_embedding_metadata WHERE id = 1"
            )
            metadata = cursor.fetchone()
            cursor.execute(
                "SELECT dog_id, image_name, image_hash, embedding, "
                "embedding_dimension, model FROM dog_embeddings ORDER BY dog_id"
            )
            dogs = [dict(row) for row in cursor.fetchall()]

    if metadata is None:
        return {"dogs": []}

    return {
        "model": metadata["model"],
        "preprocessing": metadata["preprocessing"],
        "embedding_dimension": metadata["embedding_dimension"],
        "dogs": dogs,
    }


def save_database(database: Dict) -> None:
    dogs: List[Dict] = database.get("dogs", [])
    with psycopg.connect(connection_string()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO dog_embedding_metadata "
                "(id, model, preprocessing, embedding_dimension) VALUES (1, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET model = EXCLUDED.model, "
                "preprocessing = EXCLUDED.preprocessing, "
                "embedding_dimension = EXCLUDED.embedding_dimension",
                (
                    database["model"],
                    database["preprocessing"],
                    database["embedding_dimension"],
                ),
            )
            cursor.execute("DELETE FROM dog_embeddings")
            for dog in dogs:
                cursor.execute(
                    "INSERT INTO dog_embeddings "
                    "(dog_id, image_name, image_hash, embedding, embedding_dimension, model) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        dog["dog_id"],
                        dog["image_name"],
                        dog["image_hash"],
                        dog["embedding"],
                        dog["embedding_dimension"],
                        dog["model"],
                    ),
                )
