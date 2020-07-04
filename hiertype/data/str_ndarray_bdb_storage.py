from hiertype.data.bdb_storage import BerkeleyDBStorage
import numpy as np
from bsddb3 import db
import msgpack
import msgpack_numpy


class StringNdArrayBerkeleyDBStorage(BerkeleyDBStorage[str, np.ndarray]):

    def __init__(self, kvs):
        super(StringNdArrayBerkeleyDBStorage, self).__init__(kvs)

    def encode_key(self, k: str) -> bytes:
        return k.encode()

    def encode_value(self, v: np.ndarray) -> bytes:
        return msgpack.packb(v, default=msgpack_numpy.encode)

    def decode_key(self, k: bytes) -> str:
        return k.decode()

    def decode_value(self, v: bytes) -> np.ndarray:
        return msgpack.unpackb(v, object_hook=msgpack_numpy.decode)

    @classmethod
    def open(cls, file: str, db_kind=db.DB_BTREE, mode: str = 'r'):
        kvs = db.DB()
        db_mode = {
            'r': db.DB_DIRTY_READ,
            'w': db.DB_CREATE
        }[mode]
        kvs.open(file, None, db_kind, db_mode)
        return cls(kvs)
