import inspect
import os
from redis import Redis
from datetime import datetime, timezone
from typing import Any, List, Optional
from memory_agent.memory_log import get_logger


class MemoryRedisCacheRetriever:
    """
    Memory retriever that uses Redis to cache results.
    This class is used to retrieve cached results from Redis.
    """

    redis_conn: Redis
    key_search: str = "filemeta"
    logger = get_logger(name="MemoryRedisCacheRetriever",
                        loki_url=os.getenv("LOKI_URL"))
    host_persistence_config: dict[str, Any] = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "decode_responses": True
    }

    def __init__(self, **kwargs):
        """
        Initialize the MemoryRedisCacheRetriever with
        Redis connection parameters.
        Args:
            **kwargs: Additional parameters for Redis connection, such as:
                - key_search (str): Key prefix for caching.
                - host_persistence_config (dict): Redis connection config.
        """
        self.host_persistence_config = kwargs.get(
            "host_persistence_config",
            self.host_persistence_config
        )
        self.key_search = kwargs.get("key_search", self.key_search)
        self.redis_conn = Redis(**self.host_persistence_config)

    def add_cache(self, data: dict[str, Any] = {}):
        """
        Add file metadata to Redis.

        Args:
            data (dict[str, Any]): A dictionary containing file metadata.
                Expected keys are 'file_name', 'ingested', and 'ts'.
        """
        update_at = datetime.now(timezone.utc).isoformat()
        key = f"{self.key_search}${data.get('file_name', '')}"
        file_name = data.get("file_name", None)
        if not file_name:
            raise ValueError("file_name is required in data")

        data = {
            "file_name": file_name,
            "update_at": update_at,
            "ingested": data.get("ingested", 0),
            "error": data.get("error", 0)
        }

        return self.redis_conn.hset(key, mapping=data)

    async def get_cache(self) -> List[dict[str, Any]]:
        """
        Load file metadata from Redis cache.
        Returns:
            list[dict[str, Any]]: A list of dictionaries
                containing file metadata.
        """
        pattern = f"{self.key_search}$*"
        keys = await self.redis_conn.keys(pattern)

        rows: list[dict[str, Any]] = []
        if not keys:
            return rows

        for key in keys:
            row = await self.get_cache_by(key)
            if row:
                rows.append(row)
        return rows

    async def get_cache_by(self, file_name: str) -> Optional[dict[str, Any]]:
        """
        Retrieve file metadata from Redis by file_name.

        Args:
            file_name (str): The name of the file to retrieve metadata for.

        Returns:
            Optional[dict[str, Any]]: The metadata dictionary
                if found, else None.
        """
        data = self.redis_conn.hgetall(f"{self.key_search}${file_name}")

        if inspect.isawaitable(data):
            data = await data

        if not data:
            return None

        return {
            "file_name": data.get("file_name"),
            "update_at": data.get("update_at"),
            "ingested": data.get("ingested"),
            "error": data.get("error", 0)
        }

    async def delete_cache_by(self, file_name: str) -> bool:
        """
        Delete file metadata from Redis by file_name.

        Args:
            file_name (str): The name of the file to delete metadata for.

        Returns:
            bool: True if the record was deleted, False otherwise.
        """
        key = f"{self.key_search}${file_name}"
        result = await self.redis_conn.delete(key)  # type: ignore
        return result > 0

    async def delete_cache(self) -> bool:
        """
        Delete all file metadata from Redis cache.

        Returns:
            bool: True if the cache was cleared, False otherwise.
        """
        try:
            pattern = f"{self.key_search}$*"
            keys = self.redis_conn.keys(pattern)

            if inspect.isawaitable(keys):
                keys = await keys

            if not keys:
                return False

            result = await self.redis_conn.delete(*keys)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error deleting cache: {str(e)}")
            return False

    async def update_cache_by(
        self,
        file_name: str,
        updates: dict[str, Any]
    ) -> bool:
        """
        Update file metadata in Redis by file_name.

        Args:
            file_name (str): The name of the file to update metadata for.
            updates (dict[str, Any]): The fields to update.

        Returns:
            bool: True if the record was updated, False otherwise.
        """
        key = f"{self.key_search}${file_name}"
        # Optionally update the update_at timestamp
        updates["update_at"] = datetime.now(timezone.utc).isoformat()
        result = self.redis_conn.hset(key, mapping=updates)
        if inspect.isawaitable(result):
            result = await result
        return True
