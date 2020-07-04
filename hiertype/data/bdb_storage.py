from bsddb3 import db
import numpy as np
from typing import *
from abc import abstractmethod

K = TypeVar('K')
V = TypeVar('V')


class BerkeleyDBStorage(Generic[K, V], MutableMapping[K, V]):
    """
    A high-performance key-value storage on disk, powered by BerkeleyDB.
    """

    def __init__(self, kvs: db.DB):
        self.kvs = kvs

    @abstractmethod
    def encode_key(self, k: K) -> bytes:
        pass

    @abstractmethod
    def encode_value(self, v: V) -> bytes:
        pass

    @abstractmethod
    def decode_key(self, k: bytes) -> K:
        pass

    @abstractmethod
    def decode_value(self, v: bytes) -> V:
        pass

    def __getitem__(self, k: K) -> V:
        return self.decode_value(self.kvs.get(self.encode_key(k)))

    def __setitem__(self, k: K, v: V) -> None:
        self.kvs.put(self.encode_key(k), self.encode_value(v))

    def __delitem__(self, k: K) -> None:
        self.kvs.delete(self.encode_key(k))

    def __len__(self) -> int:
        return self.kvs.stat()['ndata']

    def items(self) -> Iterator[Tuple[K, V]]:
        cursor = self.kvs.cursor()
        entry = cursor.first()
        while entry:
            raw_key, raw_value = entry
            yield self.decode_key(raw_key), self.decode_value(raw_value)
            entry = cursor.next()

    def __iter__(self) -> Iterator[V]:
        for _, v in self.items():
            yield v

    def close(self) -> None:
        self.kvs.close()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __enter__(self):
        return self
