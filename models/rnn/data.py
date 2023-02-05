import pandas as pd
import jax.numpy as jnp
import typing as t
import functools as ft
import metadata
import jax

from jax import random


Batch = t.Mapping[str, jnp.ndarray]

class DataLoader:
    def __init__(
        self, path: str, 
        split_ratio: float, 
        batch_size: int,
        block_size: int,
        seed
    ) -> None:
        self.path = path
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.block_size = block_size
        self.seed = seed
        self.pd = pd.DataFrame()

        self.d_train, self.d_val = self.load()

    def get_data(self, split: str) -> t.Iterator[Batch]:
        data = self.d_train if split == "train" else self.d_val
        ixs = random.randint(next(self.seed), (self.batch_size, ), 0, len(data) - self.block_size)
        x = jnp.stack([data[i:i+self.block_size] for i in ixs])
        y = jnp.stack([data[i+1:i+self.block_size+1] for i in ixs])
        return {'input': x, 'target': y}

    def load(self) -> t.Tuple[jnp.array, jnp.array]:
        self.df = pd.read_pickle(self.path)
        _map = metadata.NODE_IDS_TO_LABELS_MAPPING

        _edges = jnp.array([[]])

        for x, row in enumerate(metadata.ADJACENCY_MATRIX):
            for y, _ in enumerate(row):
                if metadata.ADJACENCY_MATRIX[x][y]:
                    _edges = jnp.append(_edges,
                        self.get_data_for_link(_map[x], _map[y]))

        edges = self.encode(
                    _edges.reshape((metadata.NUM_EDGES, 
                            int(len(_edges)/metadata.NUM_EDGES))))

        n = int(self.split_ratio * edges.shape[1])
        d_train = edges[0][:n]
        d_val = edges[0][n:]
        return d_train, d_val

    @staticmethod
    def encode(vec: jnp.array) -> jnp.array:
        return jnp.log(vec)

    @staticmethod
    @jax.jit
    def to_bytes(arr):
        """Converts an array of uint32 into an array of bytes in little endian"""

        @ft.partial(jax.vmap,in_axes=(None,0))
        def _to_bytes(arr, byte_index:int):
            mask = 0xFF << (8 * byte_index)
            b = (arr & mask) >> 8 * byte_index
            return  b.astype(jnp.uint8)

        return _to_bytes(arr,jnp.arange(4))

    def get_data_for_link(self, src: str, dst: str) -> jnp.array:
        ss: pd.Series = self.df[(self.df["src_host"] == src) & (self.df["dst_host"] == dst)]
        return ss.incoming_rate_avg.to_numpy()