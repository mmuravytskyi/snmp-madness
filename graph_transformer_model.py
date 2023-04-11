import jax.numpy as jnp
import haiku as hk
import typing as t

import dataclasses
import jraph
import jax


def layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to x with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


@dataclasses.dataclass
class GraphAttention(hk.Module):
    num_layers: int
    num_heads: int
    key_size: int
    alibi_mask: jnp.array
    widening_factor: int = 4
    name: t.Optional[str] = None

    @hk.transparent
    def _linear_projection(
            self,
            x: jnp.ndarray,
            head_size: int,
            name: t.Optional[str] = None,
    ) -> jnp.ndarray:
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        y = hk.Linear(self.num_heads * head_size, w_init=initializer, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def attention_query_fn(self, node_feat: jnp.array) -> jnp.array:
        """Function that generates attention queries from sender node features"""
        return layer_norm(node_feat)

    def attention_logit_fn(
            self,
            sender_feat: jnp.array,
            receiver_feat: jnp.array,
            edge_feat: jnp.array
    ) -> jnp.array:
        """Function that converts attention queries into logits for softmax attention"""

        # TODO: check if the axis is T
        query_feat = jnp.concatenate((sender_feat, receiver_feat), axis=-1)

        # our Value, Key and Query all have the same size :: K=V=Q
        query_heads = self._linear_projection(query_feat, self.key_size, "query")  # [T', H, Q=K]
        key_heads = self._linear_projection(query_feat, self.key_size, "key")  # [T, H, K]
        # value_heads = self._linear_projection(query_feat, self.key_size, "value")  # [T, H, V]

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = self.alibi_mask + attn_logits / jnp.sqrt(self.key_size)

        seq_len = sender_feat.shape[1]
        # masking future value for autoregressive predictions
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
        attn_logits = jnp.where(causal_mask, attn_logits, -1e30)
        return attn_logits

    def node_update_fn(self, node_feat: jnp.array) -> jnp.array:
        """function that updates the aggregated messages"""
        _, seq_len, model_size = node_feat.shape
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

        dense_block = hk.Sequential([
            hk.Linear(self.widening_factor * model_size, w_init=initializer),
            jax.nn.gelu,
            hk.Linear(model_size, w_init=initializer),
        ])
        return dense_block(layer_norm(node_feat))

    def __call__(
            self,
            embedded_graph: jraph.GraphsTuple  # [B, Edg, T x Emb, D]
    ) -> jraph.GraphsTuple:
        
        def _ApplyGAT(graph):
            """Applies a Graph Attention layer."""
            nodes, edges, receivers, senders, _, _, _ = graph
            # Equivalent to the sum of n_node, but statically known.
            try:
                sum_n_node = nodes.shape[0]
            except IndexError:
                raise IndexError('GAT requires node features')  # pylint: disable=raise-missing-from

            # First pass nodes through the node updater.
            nodes_norm = self.attention_query_fn(nodes)
            # We compute the softmax logits using a function that takes the
            # embedded sender and receiver attributes.
            sent_attributes = nodes_norm[senders]  # [Edg x B, T, D]
            received_attributes = nodes_norm[receivers]  # [Edg x B, T, D]
            softmax_logits = self.attention_logit_fn(
                sent_attributes, received_attributes, edges)

            # Compute the softmax weights on the entire tree.
            attn_weights = jraph.segment_softmax(softmax_logits, segment_ids=receivers,
                                                 num_segments=sum_n_node)  # [Edg x B, H, T, T']

            values_heads = self._linear_projection(
                sent_attributes, self.key_size, "value")  # [Edg x B, T, H, V]

            # Weight the values by the attention and flatten the head vectors.
            messages = jnp.einsum("...htT,...Thd->...thd", attn_weights, values_heads)
            messages = jnp.reshape(messages, (messages.shape[0], messages.shape[1], -1))  # [T', H*V]

            # Aggregate messages to nodes.
            nodes_attn = jraph.segment_sum(messages, receivers, num_segments=sum_n_node)
            # skip connection
            nodes = nodes + nodes_attn

            # Apply an update function to the aggregated messages.
            nodes_dense = self.node_update_fn(nodes_attn)
            # skip connection
            nodes = nodes + nodes_dense
            nodes = layer_norm(nodes)

            return graph._replace(nodes=nodes)

        return _ApplyGAT(embedded_graph)


@dataclasses.dataclass
class GraphTransformer(hk.Module):
    # dropout_rate: float
    model_size: int
    vocab_size: int
    graph_attention: GraphAttention
    embedding_size: int = 4
    name: t.Optional[str] = None

    def __call__(
            self,
            input_graph: jraph.GraphsTuple,  # [B x Edg, T x Emb]
    ) -> jraph.GraphsTuple:  # [B x Edg, T x Emb]

        _, seq_len = input_graph.nodes.shape

        # embed byte values in the nodes
        # initializers parametrizes as suggested in https://arxiv.org/abs/2201.11990 :: std = sqrt(1/(D*3))
        embed_init = hk.initializers.RandomNormal(stddev=0.072)
        token_embedding_map = hk.Embed(
            self.vocab_size, embed_dim=self.model_size, w_init=embed_init)
        graph_embedded = jraph.GraphMapFeatures(
            embed_node_fn=token_embedding_map
        )(input_graph)  # [B x Edg, T x Emb, D]

        # add byte position embeddings
        _byte_position_embeddings = hk.get_parameter(
            'byte_position_embeddings', [self.embedding_size, self.model_size], init=embed_init)
        byte_position_embeddings = _byte_position_embeddings
        for _ in range((seq_len // self.embedding_size) - 1):
            byte_position_embeddings = jnp.concatenate(
                (byte_position_embeddings, _byte_position_embeddings), axis=0)
        graph_embedded = jraph.GraphMapFeatures(
            embed_node_fn=lambda x: x + byte_position_embeddings
        )(graph_embedded)  # [B x Edg, T x Emb, D]

        graph_embedded = self.graph_attention(graph_embedded)
        # TODO: predict 4 bytes

        # decode the embeddings
        out_graph = jraph.GraphMapFeatures(
            embed_node_fn=hk.Linear(self.vocab_size)
        )(graph_embedded)  # [B x Edg, T x Emb]
        return out_graph
