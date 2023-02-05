# inspired by deepmind's haiku example
# https://github.com/deepmind/dm-haiku/blob/main/examples/rnn/train.py

from absl import app
from absl import flags
from absl import logging

from model import BaselineRNN, TrainingState
from data import DataLoader

import haiku as hk
import jax.numpy as jnp

import jax
import optax

DATA_FILE_PATH = flags.DEFINE_string(
    'data_path', '../../../data/samples_5m_subset_v1.pkl', '')
HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 256, '')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
SPLIT = flags.DEFINE_float('split', 0.9, '')
BLOCK_SIZE = flags.DEFINE_integer('block_size', int(24*60/5), '')
BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, '')
SEED = flags.DEFINE_integer('seed', 2137, '')
NUM_TRAINING_STEPS = flags.DEFINE_integer('training_steps', 500, '')
EVALUATION_INTERVAL = flags.DEFINE_integer('evaluation_interval', 10, '')

def make_network() -> hk.Module:
    """Defines the network architecture."""
    model = hk.DeepRNN([
        hk.LSTM(HIDDEN_SIZE.value),
        jax.nn.relu,
        hk.LSTM(HIDDEN_SIZE.value),
    ])
    return model

def make_optimizer() -> optax.GradientTransformation:
    """Defines the optimizer."""
    return optax.adam(LEARNING_RATE.value)

# @jax.jit
def sequence_loss(batch: dict) -> jnp.ndarray:
    """Unrolls the network over a sequence of inputs & targets, gets loss."""
    core = make_network()
    batch_size, _ = batch['input'].shape  # (B, T) 
    initial_state = core.initial_state(batch_size)
    _input = jnp.expand_dims(batch['input'], -1)  # (B, T, 1)

    logits, _ = hk.dynamic_unroll(core, _input, initial_state, time_major=False)  # (B, T, HIDDEN_SIZE)
    logits_batched = hk.BatchApply(hk.Linear(1))(logits)  # (B, T, 1)
    loss = jnp.mean(jnp.abs(batch['target'] - logits_batched[:, :, -1]))  # MAE
    return loss

@jax.jit
def update(state: TrainingState, batch: dict) -> TrainingState:
    """Does a step of SGD given inputs & targets."""
    _, optimizer = make_optimizer()
    _, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    gradients = jax.grad(loss_fn)(state.params, batch)
    updates, new_opt_state = optimizer(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return TrainingState(params=new_params, opt_state=new_opt_state)

def generate(context: jnp.ndarray, seq_len: int) -> jnp.array:
    """Draws samples from the model, given an initial context."""
    core = make_network()
    dense = hk.Linear(1)

    batch_size, sequence_length = context.shape  # (B, T)

    initial_state = core.initial_state(batch_size)

    _input = jnp.expand_dims(context, -1)  # (B, T, 1)
    _input = jnp.reshape(_input, (288, 1, 1))

    context_outs, state = hk.dynamic_unroll(core, _input, initial_state)
    context_outs = hk.BatchApply(dense)(context_outs)

    # Now, unroll one step at a time using the running recurrent state.
    _outs = []
    logits = context_outs[-1]
    for _ in range(seq_len - sequence_length):
        logits, state = core(logits, state)
        logits = dense(logits)
        _outs.append(logits)

    return jnp.concatenate([context_outs, jnp.stack(_outs)])

def main(_):
    rng = hk.PRNGSequence(SEED.value)
    data = DataLoader(
        path = DATA_FILE_PATH.value,
        split_ratio = SPLIT.value,
        batch_size = BATCH_SIZE.value,
        block_size = BLOCK_SIZE.value,
        seed = rng
    )
    logging.info("Model intialised")

    data_train = data.get_data("train")

    init_params_fn, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    initial_params = init_params_fn(next(rng), data_train)
    opt_init, _ = make_optimizer()
    initial_opt_state = opt_init(initial_params)

    # de facto initial state
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    loss_fn = jax.jit(loss_fn)
    train_loss_all = jnp.array([])
    eval_loss_all = jnp.array([])

    for step in range(NUM_TRAINING_STEPS.value):
        train_batch = data.get_data("train")
        state = update(state, train_batch)

        if step % EVALUATION_INTERVAL.value == 0:
            eval_batch = data.get_data("eval")
            train_loss = loss_fn(state.params, train_batch)
            eval_loss = loss_fn(state.params, eval_batch)
            train_loss_all = jnp.append(train_loss_all, train_loss)
            eval_loss_all = jnp.append(eval_loss_all, eval_loss)
            logging.info(f"Step: {step}, train loss: {float(train_loss):.3f}," \
            f" valid loss: {float(eval_loss):.3f}")

if __name__ == "__main__":
    app.run(main)