import pandas as pd
import jax.numpy as jnp
import haiku as hk
import metadata
import typing
import jax


Batch = typing.Mapping[str, jnp.array]


class DataIter:
    def __init__(self, f, data):
        self.edge_data = data
        self.func = f

    def __next__(self):
        return self.func(self.edge_data)


class SimpleDataLoader:
    def __init__(self,
                 data_path: str,
                 block_size: int,
                 batch_size: int,
                 emb_size: int,
                 split: float,
                 normalize: bool = True,
                 log: bool = True,
                 shift: bool = False,
                 batch_first: bool = False):
        if data_path.endswith(".pkl"):
            self.df = pd.read_pickle(data_path)
        elif data_path.endswith(".csv"):
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError

        self.block_size = block_size
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.split = split
        self.normalize = normalize
        self.log = log
        self.shift = shift
        self.batch_first = batch_first
        self.seed = hk.PRNGSequence(2137)

        self.connections = metadata.CONNECTIONS

        self.edges_data = []
        for conn in self.connections:
            link_data = self.get_data_for_link(conn[0], conn[1])
            self.edges_data.append(link_data)

        # For each src host take average incoming value except `cyfronet` and `ms`
        # where outgoing value from the neighbouring node is used.

        self.connections.append(("ms", "b6"))
        ms_b6 = self.df[(self.df["src_host"] == "b6") & (self.df["dst_host"] == "ms")].outgoing_rate_avg.to_numpy()
        self.edges_data.append(ms_b6)

        self.connections.append(("cyfronet", "uci"))
        cyfronet_uci = self.df[(self.df["src_host"] == "uci") &
                               (self.df["dst_host"] == "cyfronet")].outgoing_rate_avg.to_numpy()
        self.edges_data.append(cyfronet_uci)

        self.connections.append(("cyfronet", "ftj"))
        ftj_cyfronet = self.df[(self.df["src_host"] == "ftj") &
                               (self.df["dst_host"] == "cyfronet")].outgoing_rate_avg.to_numpy()
        self.edges_data.append(ftj_cyfronet)

        #  transform to a Jax NumPy array
        self.edges_data = jnp.array(self.edges_data)  # (18, NUM_SAMPLES)

        if self.log:
            # we add 1 to avoid -Inf
            self.edges_data = jnp.log(self.edges_data + 1)

        self.edges_metadata = [
            {'mean': s.mean(), 'std': s.std()} for s in self.edges_data
        ]

        if self.normalize:
            self.edges_data = jnp.apply_along_axis(self.fn_normalize, 0, self.edges_data)

        if self.shift:
            self.edges_data = jnp.apply_along_axis(self.fn_shift, 0, self.edges_data)

        split_idx = int(self.split * self.edges_data.shape[1])
        self.edges_train_data = self.edges_data[:, :split_idx]
        self.edges_test_data = self.edges_data[:, split_idx:]

        del self.edges_data

    @staticmethod
    def fn_normalize(arr: jnp.array):
        return (arr - arr.mean())/arr.std()

    @staticmethod
    def fn_shift(arr: jnp.array):
        return jnp.array(arr*25000 + 100000, dtype=jnp.uint32)

    def get_data_for_link(self, src: str = None, dst: str = None) -> jnp.array:
        ss: pd.Series = self.df[(self.df["src_host"] == src) & (self.df["dst_host"] == dst)].incoming_rate_avg
        return ss.to_numpy()

    def get_graph_data_iter(self, split: str) -> DataIter:
        assert split in ("train", "test")

        if split == "train":
            data = self.edges_train_data
        else:
            data = self.edges_test_data

        return DataIter(f=self.get_graph_batch, data=data)

    def get_data_iter(self, split: str) -> DataIter:
        assert split in ("train", "test")

        # here we have flattened data for simple RNNs
        if split == "train":
            data = jnp.ravel(self.edges_train_data)
        else:
            data = jnp.ravel(self.edges_test_data)

        return DataIter(f=self.get_batch, data=data)

    def get_batch(self, data: jnp.array) -> Batch:
        ixs = jax.random.randint(next(self.seed), (self.batch_size, ), 0, len(data) - self.block_size)
        x = jnp.stack([data[i:i+self.block_size] for i in ixs])
        y = jnp.stack([data[i+1:i+self.block_size+1] for i in ixs])
        if self.batch_first:
            return {'input': x, 'target': y}  # Batch
        else:
            return {'input': x.T, 'target': y.T}  # Batch

    def get_graph_batch(self, data: jnp.array) -> Batch:
        _len = data.shape[-1]
        ixs = jax.random.randint(next(self.seed), (self.batch_size, ), 0, _len - self.block_size)
        x = jnp.stack([data[:, i:i+self.block_size] for i in ixs])
        y = jnp.stack([data[:, i+1:i+self.block_size+1] for i in ixs])
        return {'input': x, 'target': y}  # Batch


if __name__ == "__main__":
    sdl = SimpleDataLoader("../data/samples_5m_subset_v1.pkl", 288, 8, 1, 0.85, batch_first=True)

    train_di = sdl.get_graph_data_iter("train")
    d = next(train_di)
    print("Graph shape:")
    print(d['input'].shape)

    train_di = sdl.get_data_iter("train")
    d = next(train_di)
    print("Regular shape:")
    print(d['input'].shape)
