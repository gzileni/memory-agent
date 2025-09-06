from typing import (
    List,
    Optional,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

REDIS_KEY_SEPARATOR = "$"


def _parse_configurable(
    config: RunnableConfig
) -> tuple[str, str, str | None]:
    """
    Extract the configurable part of a RunnableConfig.

    Args:
        config (RunnableConfig): The configuration to extract from.

    Returns:
        dict: The configurable part of the configuration.
    """
    configurable: dict = config.get("configurable", {})
    if not isinstance(configurable, dict):
        raise ValueError("Expected 'configurable' to be a dictionary")
    thread_id = configurable["thread_id"]
    if "checkpoint_ns" in configurable:
        checkpoint_ns = configurable["checkpoint_ns"]
    else:
        # Default namespace if not provided
        checkpoint_ns = "default"
    checkpoint_id = get_checkpoint_id(config)
    if checkpoint_id is None:
        checkpoint_id = ""
    return thread_id, checkpoint_ns, checkpoint_id


def _make_redis_checkpoint_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str
) -> str:
    """
    Generate a Redis key for a checkpoint.

    Args:
        thread_id (str): The thread identifier.
        checkpoint_ns (str): The namespace of the checkpoint.
        checkpoint_id (str): The unique identifier of the checkpoint.

    Returns:
        str: The generated Redis key for the checkpoint.
    """
    return REDIS_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_redis_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: Optional[int],
) -> str:
    """
    Generate a Redis key for checkpoint writes.

    Args:
        thread_id (str): The thread identifier.
        checkpoint_ns (str): The namespace of the checkpoint.
        checkpoint_id (str): The unique identifier of the checkpoint.
        task_id (str): The task identifier.
        idx (Optional[int]): The index of the write, if applicable.

    Returns:
        str: The generated Redis key for the checkpoint writes.
    """
    if idx is None:
        return REDIS_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )

    return REDIS_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_redis_checkpoint_key(redis_key: str) -> dict:
    """
    Parse a Redis checkpoint key into its components.

    Args:
        redis_key (str): The Redis key to parse.

    Returns:
        dict: A dictionary containing the parsed components:
            - thread_id (str): The thread identifier.
            - checkpoint_ns (str): The namespace of the checkpoint.
            - checkpoint_id (str): The unique identifier of the checkpoint.

    Raises:
        ValueError: If the key does not start with 'checkpoint'.
    """
    namespace, thread_id, checkpoint_ns, checkpoint_id = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
    """
    Parse a Redis checkpoint writes key into its components.

    Args:
        redis_key (str): The Redis key to parse.

    Returns:
        dict: A dictionary containing the parsed components:
            - thread_id (str): The thread identifier.
            - checkpoint_ns (str): The namespace of the checkpoint.
            - checkpoint_id (str): The unique identifier of the checkpoint.
            - task_id (str): The task identifier.
            - idx (str): The index of the write.

    Raises:
        ValueError: If the key does not start with 'writes'.
    """
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = (
        redis_key.split(REDIS_KEY_SEPARATOR)
    )
    if namespace != "writes":
        raise ValueError("Expected checkpoint key to start with 'writes'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]
) -> list:
    """
    Filter and sort Redis keys based on optional criteria.

    Args:
        keys (List[str]): The list of Redis keys to filter and sort.
        before (Optional[RunnableConfig]):
            Configuration to filter keys before a specific checkpoint ID.
        limit (Optional[int]): The maximum number of keys to return.

    Returns:
        list: The filtered and sorted list of Redis keys.
    """
    if before:
        before_checkpoint_id = before.get(
            "configurable", {}
        ).get("checkpoint_id", "")
        keys = [
            k
            for k in keys
            if (
                _parse_redis_checkpoint_key(
                    k if isinstance(k, str) else k.decode()
                )["checkpoint_id"] < before_checkpoint_id
            )
        ]

    keys = sorted(
        keys,
        key=lambda k: _parse_redis_checkpoint_key(
            k if isinstance(k, str) else k.decode()
        )["checkpoint_id"],
        reverse=True,
    )
    if limit:
        keys = keys[:limit]
    return keys


def _load_writes(
    serde: SerializerProtocol, task_id_to_data: dict[tuple[str, str], dict]
) -> list[PendingWrite]:
    """
    Deserialize pending writes.

    Args:
        serde (SerializerProtocol): The serializer protocol to use for
        deserialization.
        task_id_to_data (dict[tuple[str, str], dict]): A mapping of task IDs
        and data.

    Returns:
        list[PendingWrite]: A list of deserialized pending writes.
    """
    writes = [
        (
            task_id,
            data[b"channel"].decode(),
            serde.loads_typed((data[b"type"].decode(), data[b"value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
        if b"type" in data and b"value" in data and b"channel" in data
    ]
    return writes


def _parse_redis_checkpoint_data(
    serde: SerializerProtocol,
    key: str,
    data: dict,
    pending_writes: Optional[List[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    """
    Parse checkpoint data retrieved from Redis.

    Args:
        serde (SerializerProtocol): The serializer protocol to use for
            deserialization.
        key (str): The Redis key of the checkpoint.
        data (dict): The checkpoint data retrieved from Redis.
        pending_writes (Optional[List[PendingWrite]]):
            A list of pending writes, if any.

    Returns:
        Optional[CheckpointTuple]: The parsed checkpoint tuple,
        or None if the data is empty.
    """
    if not data:
        return None

    parsed_key = _parse_redis_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed(
        (
            data[b"type"].decode(),
            data[b"checkpoint"]
        )
    )
    metadata = serde.loads(data[b"metadata"].decode())
    parent_checkpoint_id = data.get(b"parent_checkpoint_id", b"").decode()
    parent_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": parent_checkpoint_id,
        }
    }
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes
    )
