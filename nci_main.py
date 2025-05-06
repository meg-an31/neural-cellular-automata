import tensorflow as tf
import numpy as np
import matplotlib as plt

WIDTH, HEIGHT = 32, 32
CHANNEL_N = 48 # 3 visible channels; the cell's RGB values. the rest are internal - up to the network to determine.
IMG_FILE_NAME = 'red_circle.png'
UPDATES_PER_TRAIN = 32 # number of iterations to make per train
BATCH_SIZE = 8

def get_img(path):
    img = plt.pyplot.imread(path)[..., :3]  # RGB only
    img = tf.image.resize(img, [HEIGHT, WIDTH])
    return tf.convert_to_tensor(img, dtype=tf.float32)

target_img = get_img(IMG_FILE_NAME)

# Normalize RGB to 0–1
target_img = target_img[None, ...]  # Add batch dim (1, H, W, 3)


class NCA(tf.keras.Model):
    def __init__(self, channel_n=CHANNEL_N):
        super().__init__()
        self.channel_n = channel_n

        # Internal layer
        self.dense128 = tf.keras.layers.Conv2D(
            128, 3, padding='same', activation='relu',
            kernel_initializer=tf.zeros_initializer()
        )

        # Output layer: maps features to delta state (residual)
        self.denseOut = tf.keras.layers.Conv2D(
            channel_n, 1, activation=None,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1)
        )

        self.dense = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128, 1, activation='relu'),
    tf.keras.layers.Conv2D(channel_n, 1, kernel_initializer=tf.random_normal_initializer(stddev=0.007))
])

    # i want to simulate the fact that cells rely upon chemical gradients
    # so i take the "sobel" filter of the surrounding cells to
    # determine the knowledge of the outside world
    # sobel: gives approximation of x and y gradients
    def perceive(self, x):
        sobel_x = tf.constant([[-1, 0, +1],
                                     [-2, 0, +2],
                                     [-1, 0, +1]], dtype=tf.float32)
        sobel_y = tf.transpose(sobel_x)

        sobel_x = sobel_x[:, :, None, None]  # Shape (3, 3, 1, 1)
        sobel_y = sobel_y[:, :, None, None]

        # apply to each channel
        # apply the sobel matrix to each channel (mixed)
        filters_x = tf.repeat(sobel_x, x.shape[-1], axis=2)  # (3, 3, channels, 1)
        filters_y = tf.repeat(sobel_y, x.shape[-1], axis=2)
        # apply the sobel matrix to each channel separately!
        # this creates the 48 sized output channel
        grad_x = tf.nn.depthwise_conv2d(x, filters_x, strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.depthwise_conv2d(x, filters_y, strides=[1, 1, 1, 1], padding='SAME')

        # concat original state + x-gradient + y-gradient
        return tf.concat([x, grad_x, grad_y], axis=-1)  # (batch, h, w, channels*3)

    def call(self, x, alive_mask=None):
        # compute change of state: only using 3 layers!
        dx = self.perceive(x)
        #dx = self.dense128(x)
        dx = self.dense(dx)

        if alive_mask is not None:
            dx *= alive_mask  # Apply stochastic mask
            x = x + dx
            x *= alive_mask
        else:
            x = x + dx
        return x


def seed_state(batch_size):
    # Create a batch of zero state grids
    state = tf.zeros([batch_size, HEIGHT, WIDTH, CHANNEL_N], dtype=tf.float32)

    # Set center cell of each batch element to "alive"
    center = [HEIGHT//2, WIDTH//2]
    updates = tf.constant([1.0] + [0.0]*(CHANNEL_N - 1), dtype=tf.float32)

    for i in range(batch_size):
        state = tf.tensor_scatter_nd_update(
            state,
            indices=[[i, center[0], center[1]]],
            updates=[updates]
        )
    return state

def get_alive_mask(x):
    # a cell is "alive" if life channel > threshold
    return tf.cast(tf.nn.relu(x[..., 0:1]) > 0.1, tf.float32) # all tensor dims except the last one

def compute_loss(state, target_img):
    # the last 3 channels are the expected rgb values
    predicted_rgb_vals = tf.clip_by_value(state[..., -3:], 0.0, 1.0)

    # in the live cells only, get the loss function for how accurate the colour is
    loss = tf.reduce_mean((predicted_rgb_vals - target_img)**2 * get_alive_mask(state))
    return loss

nca = NCA()
optimizer = tf.keras.optimizers.Adam(1e-3)

def train_step():
    with tf.GradientTape() as tape:
        # as the first random val
        x = seed_state(BATCH_SIZE)

        for i in range(UPDATES_PER_TRAIN):
            # apply stochastic update mask (randomly drop some updates)
            mask = tf.cast(tf.random.uniform([BATCH_SIZE, HEIGHT, WIDTH, 1]) > 0.5, tf.float32)
            x = nca(x, update_mask=mask)

        loss = compute_loss(x, target_img)

    # backpropagation
    grads = tape.gradient(loss, nca.trainable_variables)
    optimizer.apply_gradients(zip(grads, nca.trainable_variables))
    return loss, x

# ------------------------
# Run Training
# ------------------------

for step in range(1000):
    loss, x_out = train_step()
    print(f"Step {step}, Loss: {loss.numpy():.4f}")

    if (step % 50 == 0) and step > 0:
        print(f"Loss: {loss.numpy():.4f}")
        # Visualize one sample in batch
        vis = x_out[0, ..., -3:].numpy()
        plt.pyplot.imshow(np.clip(vis, 0, 1))
        #plt.pyplot.title(f"Step {step}")
        plt.pyplot.axis("off")
        plt.pyplot.show()