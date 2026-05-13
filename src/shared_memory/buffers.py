import json
import struct
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from src.common.typing_aliases import ArrayAny

HEADER_STRUCT = struct.Struct("<QQ")
HEADER_SIZE = HEADER_STRUCT.size

JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]


def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _unregister_shared_memory(segment: shared_memory.SharedMemory) -> None:
    try:
        resource_tracker.unregister(segment.name, "shared_memory")
    except Exception:
        pass


def _to_jsonable(value: object) -> JsonValue:
    if isinstance(value, np.ndarray):
        return cast(JsonValue, value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _segment_buffer(segment: shared_memory.SharedMemory) -> memoryview:
    buffer = segment.buf
    if buffer is None:
        raise RuntimeError(f"Shared memory buffer is unavailable for {segment.name}")
    return buffer


def encode_json_payload(payload: object) -> bytes:
    return json.dumps(_to_jsonable(payload), separators=(",", ":")).encode("utf-8")


def measure_json_payload_bytes(payload: object) -> int:
    return len(encode_json_payload(payload))


@dataclass(frozen=True)
class SharedArrayDescriptor:
    name: str
    shape: Tuple[int, ...]
    dtype: str

    def to_payload(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "SharedArrayDescriptor":
        if "name" not in payload or "shape" not in payload or "dtype" not in payload:
            raise ValueError("Shared array payload must contain name, shape, and dtype")
        shape = payload["shape"]
        if not isinstance(shape, Sequence):
            raise ValueError("Shared array shape must be a sequence")
        return cls(
            name=str(payload["name"]),
            shape=_normalize_shape(shape),
            dtype=str(payload["dtype"]),
        )


class SharedArrayPool:
    def __init__(self, *, prefix: str, slot_capacity_bytes: int, num_slots: int) -> None:
        if slot_capacity_bytes <= 0:
            raise ValueError("slot_capacity_bytes must be > 0")
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")

        self._slot_capacity_bytes = int(slot_capacity_bytes)
        self._segments: List[shared_memory.SharedMemory] = []
        for slot in range(num_slots):
            self._segments.append(
                shared_memory.SharedMemory(
                    name=f"{prefix}_{slot}",
                    create=True,
                    size=self._slot_capacity_bytes,
                )
            )

    def write(self, array: ArrayAny, *, slot_index: int) -> SharedArrayDescriptor:
        segment = self._segments[int(slot_index) % len(self._segments)]
        contiguous = np.ascontiguousarray(array)
        nbytes = int(contiguous.nbytes)
        if nbytes > self._slot_capacity_bytes:
            raise ValueError(
                f"Array of {nbytes} bytes exceeds slot capacity {self._slot_capacity_bytes} for {segment.name}"
            )

        view: ArrayAny = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=_segment_buffer(segment),
        )
        view[...] = contiguous
        return SharedArrayDescriptor(
            name=segment.name,
            shape=tuple(int(dim) for dim in contiguous.shape),
            dtype=str(contiguous.dtype),
        )

    def close(self) -> None:
        for segment in self._segments:
            try:
                segment.close()
            finally:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


class SharedArrayReader:
    def __init__(self) -> None:
        self._segments: Dict[str, shared_memory.SharedMemory] = {}

    def _get_segment(self, name: str) -> shared_memory.SharedMemory:
        segment = self._segments.get(name)
        if segment is None:
            segment = shared_memory.SharedMemory(name=name, create=False)
            _unregister_shared_memory(segment)
            self._segments[name] = segment
        return segment

    def read(self, descriptor_payload: Dict[str, object]) -> ArrayAny:
        descriptor = SharedArrayDescriptor.from_payload(descriptor_payload)
        segment = self._get_segment(descriptor.name)
        array: ArrayAny = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=_segment_buffer(segment),
        )
        return np.array(array, copy=True)

    def close(self) -> None:
        for segment in self._segments.values():
            segment.close()
        self._segments.clear()


class SharedMessageBuffer:
    def __init__(self, *, name: str, size_bytes: int, create: bool) -> None:
        if size_bytes <= HEADER_SIZE:
            raise ValueError("size_bytes must be larger than the header size")
        self._name = name
        self._size_bytes = int(size_bytes)
        self._owner = bool(create)
        self._segment = shared_memory.SharedMemory(name=name, create=create, size=self._size_bytes)
        if not self._owner:
            _unregister_shared_memory(self._segment)
        self._write_version = 0
        if create:
            _segment_buffer(self._segment)[:HEADER_SIZE] = b"\x00" * HEADER_SIZE

    def write(self, payload: object) -> int:
        data = encode_json_payload(payload)
        if len(data) > self._size_bytes - HEADER_SIZE:
            raise ValueError(
                f"Serialized payload of {len(data)} bytes exceeds buffer capacity {self._size_bytes - HEADER_SIZE}"
            )

        start_version = self._write_version + 1
        end_version = self._write_version + 2
        buffer = _segment_buffer(self._segment)
        HEADER_STRUCT.pack_into(buffer, 0, start_version, len(data))
        buffer[HEADER_SIZE : HEADER_SIZE + len(data)] = data
        HEADER_STRUCT.pack_into(buffer, 0, end_version, len(data))
        self._write_version = end_version
        return end_version

    def read(self, *, last_version: Optional[int] = None) -> Tuple[int, Optional[object]]:
        buffer = _segment_buffer(self._segment)
        while True:
            version_1, length_1 = HEADER_STRUCT.unpack_from(buffer, 0)
            if version_1 == 0:
                return 0, None
            if version_1 % 2 == 1:
                continue
            if last_version is not None and version_1 == last_version:
                return version_1, None

            payload_bytes = bytes(buffer[HEADER_SIZE : HEADER_SIZE + length_1])
            version_2, length_2 = HEADER_STRUCT.unpack_from(buffer, 0)
            if version_1 == version_2 and length_1 == length_2 and version_2 % 2 == 0:
                return version_2, json.loads(payload_bytes.decode("utf-8"))

    def close(self) -> None:
        try:
            self._segment.close()
        finally:
            if self._owner:
                try:
                    self._segment.unlink()
                except FileNotFoundError:
                    pass
