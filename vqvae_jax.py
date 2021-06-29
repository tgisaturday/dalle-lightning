import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.enable_v2_behavior()

print("JAX version {}".format(jax.__version__))
print("Haiku version {}".format(hk.__version__))
print("TF version {}".format(tf.__version__))


cifar10 = tfds.as_numpy(tfds.load("cifar10", split="train+test", batch_size=-1))
del cifar10["id"], cifar10["label"]
jax.tree_map(lambda x: f'{x.dtype.name}{list(x.shape)}', cifar10)

train_data_dict = jax.tree_map(lambda x: x[:40000], cifar10)
valid_data_dict = jax.tree_map(lambda x: x[40000:50000], cifar10)
test_data_dict = jax.tree_map(lambda x: x[50000:], cifar10)

def cast_and_normalise_images(data_dict):
  """Convert images to floating point with the range [-0.5, 0.5]"""
  data_dict['image'] = (tf.cast(data_dict['image'], tf.float32) / 255.0) - 0.5
  return data_dict

train_data_variance = np.var(train_data_dict['image'] / 255.0)
print('train data variance: %s' % train_data_variance)




class ResidualStack(hk.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name=None):
    super(ResidualStack, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._layers = []
    for i in range(num_residual_layers):
      conv3 = hk.Conv2D(
          output_channels=num_residual_hiddens,
          kernel_shape=(3, 3),
          stride=(1, 1),
          name="res3x3_%d" % i)
      conv1 = hk.Conv2D(
          output_channels=num_hiddens,
          kernel_shape=(1, 1),
          stride=(1, 1),
          name="res1x1_%d" % i)
      self._layers.append((conv3, conv1))

  def __call__(self, inputs):
    h = inputs
    for conv3, conv1 in self._layers:
      conv3_out = conv3(jax.nn.relu(h))
      conv1_out = conv1(jax.nn.relu(conv3_out))
      h += conv1_out
    return jax.nn.relu(h)  # Resnet V1 style


class Encoder(hk.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name=None):
    super(Encoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._enc_1 = hk.Conv2D(
        output_channels=self._num_hiddens // 2,
        kernel_shape=(4, 4),
        stride=(2, 2),
        name="enc_1")
    self._enc_2 = hk.Conv2D(
        output_channels=self._num_hiddens,
        kernel_shape=(4, 4),
        stride=(2, 2),
        name="enc_2")
    self._enc_3 = hk.Conv2D(
        output_channels=self._num_hiddens,
        kernel_shape=(3, 3),
        stride=(1, 1),
        name="enc_3")
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)

  def __call__(self, x):
    h = jax.nn.relu(self._enc_1(x))
    h = jax.nn.relu(self._enc_2(h))
    h = jax.nn.relu(self._enc_3(h))
    return self._residual_stack(h)

class Decoder(hk.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name=None):
    super(Decoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._dec_1 = hk.Conv2D(
        output_channels=self._num_hiddens,
        kernel_shape=(3, 3),
        stride=(1, 1),
        name="dec_1")
    self._residual_stack = ResidualStack(
        self._num_hiddens,
        self._num_residual_layers,
        self._num_residual_hiddens)
    self._dec_2 = hk.Conv2DTranspose(
        output_channels=self._num_hiddens // 2,
        # output_shape=None,
        kernel_shape=(4, 4),
        stride=(2, 2),
        name="dec_2")
    self._dec_3 = hk.Conv2DTranspose(
        output_channels=3,
        # output_shape=None,
        kernel_shape=(4, 4),
        stride=(2, 2),
        name="dec_3")
    
  def __call__(self, x):
    h = self._dec_1(x)
    h = self._residual_stack(h)
    h = jax.nn.relu(self._dec_2(h))
    x_recon = self._dec_3(h)
    return x_recon
    
class VQVAEModel(hk.Module):
  def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, 
               data_variance, name=None):
    super(VQVAEModel, self).__init__(name=name)
    self._encoder = encoder
    self._decoder = decoder
    self._vqvae = vqvae
    self._pre_vq_conv1 = pre_vq_conv1
    self._data_variance = data_variance

  def __call__(self, inputs, is_training):
    z = self._pre_vq_conv1(self._encoder(inputs))
    vq_output = self._vqvae(z, is_training=is_training)
    x_recon = self._decoder(vq_output['quantize'])
    recon_error = jnp.mean((x_recon - inputs) ** 2) / self._data_variance
    loss = recon_error + vq_output['loss']
    return {
        'z': z,
        'x_recon': x_recon,
        'loss': loss,
        'recon_error': recon_error,
        'vq_output': vq_output,
    }



# Set hyper-parameters.
batch_size = 32
image_size = 32

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 100000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
# These hyper-parameters define the size of the model (number of parameters and layers).
# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
embedding_dim = 64

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = True

# This is only used for EMA updates.
decay = 0.99

learning_rate = 3e-4

# # Data Loading.
train_dataset = tfds.as_numpy(
    tf.data.Dataset.from_tensor_slices(train_data_dict)
    .map(cast_and_normalise_images)
    .shuffle(10000)
    .repeat(-1)  # repeat indefinitely
    .batch(batch_size, drop_remainder=True)
    .prefetch(-1))
valid_dataset = tfds.as_numpy(
    tf.data.Dataset.from_tensor_slices(valid_data_dict)
    .map(cast_and_normalise_images)
    .repeat(1)  # 1 epoch
    .batch(batch_size)
    .prefetch(-1))

# # Build modules.
def forward(data, is_training):
  encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
  decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
  pre_vq_conv1 = hk.Conv2D(
      output_channels=embedding_dim,
      kernel_shape=(1, 1),
      stride=(1, 1),
      name="to_vq")

  if vq_use_ema:
    vq_vae = hk.nets.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay)
  else:
    vq_vae = hk.nets.VectorQuantizer(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost)
    
  model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                     data_variance=train_data_variance)

  return model(data['image'], is_training)

forward = hk.transform_with_state(forward)
optimizer = optax.adam(learning_rate)

@jax.jit
def train_step(params, state, opt_state, data, axis_name='i'):
  def adapt_forward(params, state, data):
    # Pack model output and state together.
    model_output, state = forward.apply(params, state, None, data, is_training=True)
    loss = model_output['loss']
    return loss, (model_output, state)

  grads, (model_output, state) = (
      jax.grad(adapt_forward, has_aux=True)(params, state, data))

  grads = jax.lax.pmean(grads, axis_name)  
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return params, state, opt_state, model_output

train_losses = []
train_recon_errors = []
train_perplexities = []
train_vqvae_loss = []

rng = jax.random.PRNGKey(42)
train_dataset_iter = iter(train_dataset)
params, state = forward.init(rng, next(train_dataset_iter), is_training=True)

opt_state = optimizer.init(params)

num_devices = jax.local_device_count()
params = jax.tree_util.tree_map(lambda x: np.stack([x] * num_devices), params)
state = jax.tree_util.tree_map(lambda x: np.stack([x] * num_devices), state)
#opt_state = jax.tree_util.tree_map(lambda x: np.stack([x] * num_devices), opt_state)
def make_superbatch():
  """Constructs a superbatch, i.e. one batch of data per device."""
  # Get N batches, then split into list-of-images and list-of-labels.
  superbatch = [next(train_dataset_iter) for _ in range(num_devices)]
  # Stack the superbatches to be one array with a leading dimension, rather than
  # a python list. This is what `jax.pmap` expects as input.
  superbatch = np.stack(superbatch)

  return superbatch


for step in range(1, num_training_updates + 1):
  data = make_superbatch()

  params, state, opt_state, train_results = jax.pmap(
      train_step, axis_name='i')(params, state, opt_state, data)

  train_results = jax.device_get(train_results)
  train_losses.append(train_results['loss'])
  train_recon_errors.append(train_results['recon_error'])
  train_perplexities.append(train_results['vq_output']['perplexity'])
  train_vqvae_loss.append(train_results['vq_output']['loss'])

  if step % 100 == 0:
    print(f'[Step {step}/{num_training_updates}] ' + 
          ('train loss: %f ' % np.mean(train_losses[-100:])) +
          ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
          ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
          ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))  

