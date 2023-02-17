import pandas as pd
import jax.numpy as jnp
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
    def __init__(self, data_path: str, block_size: int, batch_size: int, emb_size: int, split: float):
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
        self.seed = jax.random.PRNGKey(2137)

        _map = metadata.NODE_IDS_TO_LABELS_MAPPING
        self.edges_to_samples_mapping = {}

        for x, row in enumerate(metadata.ADJACENCY_MATRIX):
            for y, _ in enumerate(row):

                if metadata.ADJACENCY_MATRIX[x][y]:
                    self.edges_to_samples_mapping[(_map[x], _map[y])] = \
                        self.get_data_for_link(_map[x], _map[y])

        # collect metadata
        self.metadata = {
            k: {'mean': samples.mean(), 'std': samples.std()}
            for k, samples in self.edges_to_samples_mapping.items()
        }

        # train/test split
        data_size = self.edges_to_samples_mapping[('uci', 'ftj')].shape[0]
        split_idx = int(self.split * data_size)
        self.train_ds: dict = {
            k: samples[:split_idx] for k, samples in self.edges_to_samples_mapping.items()
        }
        self.test_ds: dict = {
            k: samples[split_idx:] for k, samples in self.edges_to_samples_mapping.items()
        }
        self.unified_train_ds: jnp.array = jnp.array([sample for _, samples in
                                                      self.train_ds.items() for sample in samples])

        del self.edges_to_samples_mapping

    @staticmethod
    @jax.jit
    def normalize(arr: jnp.array):
        # here we add 1 in order to avoid -INF for values below 1
        return jnp.log(arr + 1)

    def get_data_for_link(self, src: str, dst: str) -> jnp.array:
        ss: pd.Series = self.df[(self.df["src_host"] == src) & (self.df["dst_host"] == dst)]
        return ss.incoming_rate_avg.to_numpy()

    def get_data_iter(self, split: str, src: str = None, dst: str = None) -> DataIter:
        assert split in ("train", "test")

        if split == "train":
            data = self.unified_train_ds
        else:
            assert src and dst
            data = self.test_ds[(src, dst)]

        return DataIter(f=self.get_batch, data=data)

    def get_batch(self, data: jnp.array) -> Batch:
        ixs = jax.random.randint(self.seed, (self.batch_size, ), 0, len(data) - self.block_size)
        x = jnp.stack([data[i:i+self.block_size] for i in ixs]).T
        y = jnp.stack([data[i+1:i+self.block_size+1] for i in ixs]).T
        return {'input': x, 'target': y}  # Batch


class DataLoader:
    def __init__(self, data_path: str, block_size: int, batch_size: int, emb_size: int, split: float):
        self.df = pd.read_pickle(data_path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.split = split

        _map = metadata.NODE_IDS_TO_LABELS_MAPPING
        self.edges_to_samples_mapping = {}

        for x, row in enumerate(metadata.ADJACENCY_MATRIX):
            for y, _ in enumerate(row):

                if metadata.ADJACENCY_MATRIX[x][y]:
                    self.edges_to_samples_mapping[(_map[x], _map[y])] = \
                        self.get_data_for_link(_map[x], _map[y])

        _b = self.block_size * self.batch_size
        # all the sample array are equal size, so it doesn't matter which one we choose
        num_subsets = (self.edges_to_samples_mapping[('uci', 'ftj')].shape[0] - 1) // _b
        cutoff_num = (self.edges_to_samples_mapping[('uci', 'ftj')].shape[0] - 1) % _b
        split_idx = int(self.split * num_subsets)

        # collect metadata
        self.metadata = {
            k: {'mean': samples.mean(), 'std': samples.std()}
            for k, samples in self.edges_to_samples_mapping.items()
        }

        #  encode & normalize the data
        self.edges_to_samples_mapping = {
            k: self.normalize(self.encode(samples))
            for k, samples in self.edges_to_samples_mapping.items()
        }

        #  input/target split
        self.edges_to_samples_mapping = {
            k: {'input': samples[:-1], 'target': samples[1:]}
            for k, samples in self.edges_to_samples_mapping.items()
        }

        #  rounding to batchable size & pre-batching
        self.edges_to_samples_mapping = {
            k: {
                'input': jnp.reshape(samples['input'][:-cutoff_num],
                                     (num_subsets, self.block_size, self.batch_size, self.emb_size)),
                'target': jnp.reshape(samples['target'][:-cutoff_num],
                                      (num_subsets, self.block_size, self.batch_size, self.emb_size))
            } for k, samples in self.edges_to_samples_mapping.items()
        }

        #  batching
        self.edges_to_samples_mapping = {
            k: [
                {
                    'input': samples['input'][idx], 'target': samples['target'][idx]
                } for idx in range(samples['input'].shape[0])
            ] for k, samples in self.edges_to_samples_mapping.items()
        }

        self.train_ds = {
            k: samples[:split_idx] for k, samples in self.edges_to_samples_mapping.items()
        }
        self.test_ds = {
            k: samples[split_idx:] for k, samples in self.edges_to_samples_mapping.items()
        }
        self.unified_train_ds = [sample for _, samples in self.train_ds.items() for sample in samples]

    def get_edge_data_iterator(self, src: str, dst: str, train=True, all=True):
        if train:
            if all:
                data = self.unified_train_ds
            else:
                data = self.train_ds[(src, dst)]
        else:
            data = self.test_ds[(src, dst)]
        return DataIter(data)

    @staticmethod
    def encode(arr: jnp.array):
        # here we add 1 in order to avoid -INF for values below 1
        return jnp.log(arr + 1)

    @staticmethod
    def normalize(arr: jnp.array):
        return (arr - arr.mean())/arr.std()

    def get_data_for_link(self, src: str, dst: str) -> jnp.array:
        ss: pd.Series = self.df[(self.df["src_host"] == src) & (self.df["dst_host"] == dst)]
        return ss.incoming_rate_avg.to_numpy()
