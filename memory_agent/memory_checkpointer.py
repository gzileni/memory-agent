import os
import inspect
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    List,
    Optional,
    Tuple,
)
from langchain_core.runnables import RunnableConfig
from typing import Sequence
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from redis.asyncio import Redis as AsyncRedis
from .memory_log import get_metadata, get_logger
from .memory_redis import (_make_redis_checkpoint_key,
                           _make_redis_checkpoint_writes_key,
                           _parse_redis_checkpoint_key,
                           _parse_redis_checkpoint_writes_key,
                           _filter_keys,
                           _load_writes,
                           _parse_redis_checkpoint_data,
                           _parse_configurable)


class MemoryCheckpointer(BaseCheckpointSaver):
    """
    MemoryCheckpointer provides asynchronous checkpoint storage and retrieval
    using Redis.
    This class implements an asynchronous checkpoint saver that interacts with
    a Redis database to persist, retrieve, list, and delete checkpoints and
    their associated metadata and writes.
    It is designed to be used in environments where non-blocking I/O is
    required, such as async Python applications.
    Attributes:
        conn (AsyncRedis): The asynchronous Redis connection instance.
    Methods:
        __init__(conn: AsyncRedis):
            Initialize the MemoryCheckpointer with an existing AsyncRedis
            connection.
        from_conn_info(host: str, port: int, db: int) -> AsyncIterator[
            "MemoryCheckpointer"]:
            Asynchronous context manager to create an MemoryCheckpointer from
            connection info.
        aput(config: RunnableConfig, checkpoint: Checkpoint,
             metadata: CheckpointMetadata, new_versions: ChannelVersions)
             -> RunnableConfig:
            Asynchronously save a checkpoint and its metadata to Redis.
        aput_writes(config: RunnableConfig, writes: Sequence[Tuple[str, Any]],
                    task_id: str, task_path: Optional[str] = None):
            Asynchronously store intermediate writes associated with a
            checkpoint.
        aget_tuple(config: RunnableConfig) -> Optional[CheckpointTuple]:
            Asynchronously retrieve a checkpoint tuple from Redis.
        adelete_by_thread_id(thread_id: str, checkpoint_ns: str = "",
                            filter_minutes: int = 0) -> None:
            Asynchronously delete all checkpoints for a specific thread and
            namespace, optionally filtering by age.
        alist(config: Optional[RunnableConfig], *,
              filter: Optional[dict[str, Any]] = None,
              before: Optional[RunnableConfig] = None,
              limit: Optional[int] = None
        ) -> AsyncGenerator[CheckpointTuple, None]:
            Asynchronously list checkpoint tuples from Redis, optionally
            filtered and limited.
        _aload_pending_writes(thread_id: str, checkpoint_ns: str,
                             checkpoint_id: str) -> List[PendingWrite]:
            Asynchronously load all pending writes for a specific checkpoint.
        _aget_checkpoint_key(conn, thread_id: str, checkpoint_ns: str,
                            checkpoint_id: Optional[str]) -> Optional[str]:
            Asynchronously determine the Redis key for a checkpoint,
            retrieving the latest if no ID is provided.
    Usage:
        Use MemoryCheckpointer to persist and manage checkpoints in an async
        application, ensuring non-blocking
        operations when interacting with Redis.
    """

    conn: AsyncRedis
    logger = get_logger(
        name="memory_checkpointer",
        loki_url=os.getenv("LOKI_URL"),
    )

    def __init__(self, **kwargs):
        super().__init__()
        self.conn = AsyncRedis(
            **kwargs
        )

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls,
        **kwargs
    ) -> AsyncIterator["MemoryCheckpointer"]:
        conn = None
        try:
            conn = AsyncRedis(**kwargs)
            yield MemoryCheckpointer(conn=conn)
        finally:
            if conn:
                await conn.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to Redis. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the
                checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with
                the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this
                write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id, checkpoint_ns, parent_checkpoint_id = _parse_configurable(
            config
        )
        checkpoint_id = checkpoint["id"]
        key = _make_redis_checkpoint_key(
            thread_id, checkpoint_ns, checkpoint_id
        )

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "thread_id": thread_id,
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "checkpoint_id": checkpoint_id,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id,
            "ts": datetime.now(timezone.utc).isoformat()
        }

        r = self.conn.hset(key, mapping=data)
        if inspect.isawaitable(r):
            r = await r
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: Optional[str] = None,
    ):
        """
        Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to
        the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each
                as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (Optional[str]): Optional path for the task, not used in
                this implementation.
        """
        thread_id, checkpoint_ns, checkpoint_id = _parse_configurable(config)

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                str(checkpoint_id),
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Use HSET which will overwrite existing values
                i = self.conn.hset(key, mapping=data)
                if inspect.isawaitable(i):
                    i = await i
            else:
                # Use HSETNX which will not overwrite existing values
                for field, value in data.items():
                    i = self.conn.hsetnx(key, field, value)
                    if inspect.isawaitable(i):
                        i = await i

    async def aget_tuple(
        self, config: RunnableConfig
    ) -> Optional[CheckpointTuple]:
        """
        Get a checkpoint tuple from Redis asynchronously.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the
        checkpoint with the matching thread ID and checkpoint ID is retrieved.
        Otherwise, the latest checkpoint for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the
                checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None
                if no matching checkpoint was found.
        """
        thread_id, checkpoint_ns, checkpoint_id = _parse_configurable(config)

        checkpoint_key = await self._aget_checkpoint_key(
            self.conn,
            thread_id,
            checkpoint_ns,
            checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = self.conn.hgetall(checkpoint_key)
        if inspect.isawaitable(checkpoint_data):
            checkpoint_data = await checkpoint_data

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde,
            checkpoint_key,
            checkpoint_data,
            pending_writes=pending_writes,
        )

    async def adelete_by_thread_id(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        filter_minutes: int = 0
    ) -> None:
        """
        Delete all checkpoints for a specific thread ID and namespace
        asynchronously.

        Args:
            thread_id (str): The thread identifier.
            checkpoint_ns (str): The namespace of the checkpoints to delete.
                Defaults to an empty string.
            filter_minutes (int): If greater than 0, only checkpoints older
                than this many minutes will be deleted. Defaults to 0, which
                deletes all checkpoints.
        """
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = await self.conn.keys(pattern)
        if filter_minutes == 0:
            if keys:
                await self.conn.delete(*keys)
        else:
            now = datetime.now(timezone.utc)
            for key in keys:
                data = self.conn.hgetall(key)
                if inspect.isawaitable(data):
                    data = await data

                if b"ts" in data:
                    try:
                        ts = data[b"ts"].decode()
                        ts_fixed = ts.replace("Z", "+00:00")
                        ts_dt = datetime.fromisoformat(ts_fixed)
                        if ts_dt < (now - timedelta(minutes=filter_minutes)):
                            await self.conn.delete(key)
                    except Exception as e:
                        self.logger.error(
                            f"adelete_by_thread_id - Errore su {key}: {e}",
                            get_metadata(thread_id)
                        )
                        raise e

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """
        List checkpoints from Redis asynchronously.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).

        Args:
            config (Optional[RunnableConfig]):
                Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]):
                Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]):
                If provided, only checkpoints before the specified
                checkpoint ID are returned.
                Defaults to None.
            limit (Optional[int]):
                Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]:
                An asynchronous iterator of matching checkpoint tuples.
        """
        if config is None:
            raise ValueError("Config must be provided to list checkpoints.")

        thread_id, checkpoint_ns, _ = _parse_configurable(config)
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(await self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if inspect.isawaitable(data):
                data = await data
            if data and b"checkpoint" in data and b"metadata" in data:
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = await self._aload_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                checkpoint_tuple = _parse_redis_checkpoint_data(
                    self.serde,
                    key.decode(),
                    data,
                    pending_writes=pending_writes,
                )
                if checkpoint_tuple is not None:
                    yield checkpoint_tuple

    async def _aload_pending_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str
    ) -> List[PendingWrite]:
        """
        Asynchronously load pending writes for a specific checkpoint.

        This method retrieves all pending writes associated with a given
        checkpoint from Redis, parses their keys, and deserializes their data.

        Args:
            thread_id (str): The thread identifier.
            checkpoint_ns (str): The namespace of the checkpoint.
            checkpoint_id (str): The unique identifier of the checkpoint.

        Returns:
            List[PendingWrite]: A list of deserialized pending writes.
        """
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode())
            for key in matching_keys
        ]
        task_id_to_data = {}
        for key, parsed_key in sorted(
            zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
        ):
            data = self.conn.hgetall(key)
            if inspect.isawaitable(data):
                data = await data
            task_id_to_data[(parsed_key["task_id"], parsed_key["idx"])] = data

        pending_writes = _load_writes(
            self.serde,
            task_id_to_data,
        )
        return pending_writes

    async def _aget_checkpoint_key(
        self,
        conn,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """
        Asynchronously determine the Redis key for a checkpoint.

        This method retrieves the Redis key for a specific checkpoint. If a
        checkpoint ID is provided, it generates the key directly. Otherwise,
        it retrieves the latest checkpoint key for the given thread and
        namespace.

        Args:
            conn: The Redis connection instance.
            thread_id (str): The thread identifier.
            checkpoint_ns (str): The namespace of the checkpoint.
            checkpoint_id (Optional[str]):
                The unique identifier of the checkpoint, if available.

        Returns:
            Optional[str]:
                The Redis key for the checkpoint, or None if no matching key
                is found.
        """
        if checkpoint_id:
            return _make_redis_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )

        all_keys = await conn.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(
                k.decode()
            )["checkpoint_id"],
        )
        return latest_key.decode()
