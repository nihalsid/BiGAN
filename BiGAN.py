import matplotlib
matplotlib.use('Agg')

import os
import shutil
import time
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

INPUT_DIM = 28 * 28
LATENT_DIM = 64
BATCH_SIZE = 32
LOG_FREQUENCY = 1000
LEARNING_RATE = 1e-3


def safe_log(x):
    return tf.log(x + 1e-8)


def inference_discriminator(placeholder_input, placeholder_latent, reuse=False):
    placeholder_disciminator_input = tf.concat((placeholder_input, placeholder_latent), 1)
    with tf.variable_scope('discriminator', reuse=reuse):
        d_net = slim.fully_connected(placeholder_disciminator_input, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='dis1')
        d_net = slim.fully_connected(d_net, 1, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='dis3')
    return d_net


def inference_generator(placeholder_latent, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        d_net = slim.fully_connected(placeholder_latent, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='gen1')
        d_net = slim.fully_connected(d_net, INPUT_DIM, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='gen3')
    return d_net


def inference_encoder(placeholder_input, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        d_net = slim.fully_connected(placeholder_input, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='enc1')
        d_net = slim.fully_connected(d_net, LATENT_DIM, activation_fn=None, reuse=reuse, scope='enc3')
    return d_net


def loss_discriminator(output_discriminator_input_p_encoder, output_discriminator_generator_p_latent):
    loss = -safe_log(output_discriminator_input_p_encoder) - safe_log(1 - output_discriminator_generator_p_latent)
    return tf.reduce_mean(loss)


def loss_generator_and_encoder(output_discriminator_input_p_encoder, output_discriminator_generator_p_latent):
    loss = -safe_log(output_discriminator_generator_p_latent) - safe_log(1 - output_discriminator_input_p_encoder)
    return tf.reduce_mean(loss)


def train(loss_exp_discriminator, loss_exp_generator_and_encoder):
    vars_d = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    vars_g_e = [var for var in tf.trainable_variables() if 'generator' in var.name or 'encoder' in var.name]
    global_step_d = tf.Variable(0, name='global_step_d', trainable=False)
    global_step_g_e = tf.Variable(0, name='global_step_g_e', trainable=False)
    optimizer_d = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    optimizer_g_e = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op_d = slim.learning.create_train_op(loss_exp_discriminator, optimizer_d, global_step=global_step_d, variables_to_train=vars_d)
    train_op_g_e = slim.learning.create_train_op(loss_exp_generator_and_encoder, optimizer_g_e, global_step=global_step_g_e, variables_to_train=vars_g_e)
    return train_op_d, train_op_g_e


def placeholders():
    placeholder_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
    placeholder_latent = tf.placeholder(tf.float32, shape=[None, LATENT_DIM])
    return placeholder_input, placeholder_latent


def fill_feed_dictionary(dataset, placeholder_input, placeholder_latent, nsamples=BATCH_SIZE):
    input_next, _ = dataset.next_batch(nsamples)
    latent_next = np.random.uniform(-1.0, 1.0, size=[nsamples, LATENT_DIM])
    return {
        placeholder_input: input_next,
        placeholder_latent: latent_next
    }


def visualize_samples(samples, step):
    plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    for j, generated_image in enumerate(samples):
        ax = plt.subplot(gs[j])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(generated_image.reshape(28, 28), cmap='Greys_r')
    if not os.path.exists('res'):
        os.makedirs('res')
    plt.savefig('res/{}.png'.format(str(step).zfill(7)), bbox_inches='tight')
    plt.close()


def run_training(max_epochs):
    dataset = input_data.read_data_sets('MNIST_data')

    with tf.Graph().as_default():
        # Create graph
        placeholder_input, placeholder_latent = placeholders()
        output_generator = inference_generator(placeholder_latent)
        output_encoder = inference_encoder(placeholder_input)
        output_discriminator_input_p_encoder = inference_discriminator(placeholder_input, output_encoder)
        output_discriminator_generator_p_latent = inference_discriminator(output_generator, placeholder_latent, reuse=True)
        loss_exp_discriminator = loss_discriminator(output_discriminator_input_p_encoder, output_discriminator_generator_p_latent)
        loss_exp_generator_encoder = loss_generator_and_encoder(output_discriminator_input_p_encoder, output_discriminator_generator_p_latent)
        train_op_discriminator, train_op_generator_encoder = train(loss_exp_discriminator, loss_exp_generator_encoder)

        # Create session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Summaries
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.summary.scalar('loss_D', loss_exp_discriminator)
        tf.summary.scalar('loss_G_E', loss_exp_generator_encoder)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./summaries', sess.graph)

        duration = 0
        print('%-10s | %-20s | %-20s | %-10s' % ('Epoch', 'Loss(D)', 'Loss(G+E)', 'Time(s)'))
        print('-' * 86)
        for step in range(max_epochs):

            # Train the neural network graph
            start_time = time.time()
            summary_d, _, loss_val_d = sess.run([merged, train_op_discriminator, loss_exp_discriminator], feed_dict=fill_feed_dictionary(dataset.train, placeholder_input, placeholder_latent))
            summary_g, _, loss_val_g = sess.run([merged, train_op_generator_encoder, loss_exp_generator_encoder], feed_dict=fill_feed_dictionary(dataset.train, placeholder_input, placeholder_latent))
            duration += (time.time() - start_time)
            summary_writer.add_summary(summary_d, step)
            summary_writer.add_summary(summary_g, step)

            # Visualize and report train stats
            if step % LOG_FREQUENCY == 0:
                print('%-10s | %-20s | %-20s | %-10s' % ('%d' % step, '%.5f' % loss_val_d, '%.5f' % loss_val_g, '%.2f' % duration))
                samples = sess.run(output_generator, feed_dict=fill_feed_dictionary(dataset.train, placeholder_input, placeholder_latent, 25))
                visualize_samples(samples, step)
                duration = 0


if __name__ == '__main__':
    if os.path.exists('summaries'):
        shutil.rmtree('summaries')
    if os.path.exists('res'):
        shutil.rmtree('res')
    run_training(50001)
