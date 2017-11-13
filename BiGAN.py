import os
import time
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from misc import leaky_relu
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

INPUT_DIM = 28 * 28
LATENT_DIM = 50
BATCH_SIZE = 100
LOG_FREQUENCY = 1000


def inference_discriminator(placeholder_input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        d_net = slim.fully_connected(placeholder_input, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='dis1')
        #d_net = slim.fully_connected(d_net, 1024, activation_fn=tf.nn.relu, reuse=reuse, scope='dis2')
        d_net = slim.fully_connected(d_net, 1, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='dis3')
    return d_net


def inference_generator(placeholder_latent, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        d_net = slim.fully_connected(placeholder_latent, 128, activation_fn=tf.nn.relu, reuse=reuse, scope='gen1')
        #d_net = slim.fully_connected(d_net, 1024, activation_fn=tf.nn.relu, reuse=reuse, scope='gen2')
        d_net = slim.fully_connected(d_net, INPUT_DIM, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='gen3')
    return d_net


def loss_discriminator(output_discriminator_dataset, output_discriminator_generated):
    loss = -tf.log(output_discriminator_dataset) - tf.log(1 - output_discriminator_generated)
    return tf.reduce_mean(loss)


def loss_generator(output_discriminator_generated):
    loss = -tf.log(output_discriminator_generated)
    return tf.reduce_mean(loss)


def train(loss_exp_discriminator, loss_exp_generator):
    vars_g = [var for var in tf.trainable_variables() if 'generator' in var.name]
    vars_d = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    global_step_d = tf.Variable(0, name='global_step_d', trainable=False)
    global_step_g = tf.Variable(0, name='global_step_g', trainable=False)
    optimizer_d = tf.train.AdamOptimizer()
    optimizer_g = tf.train.AdamOptimizer()
    train_op_d = slim.learning.create_train_op(loss_exp_discriminator, optimizer_d, global_step=global_step_d, variables_to_train=vars_d)
    train_op_g = slim.learning.create_train_op(loss_exp_generator, optimizer_g, global_step=global_step_g, variables_to_train=vars_g)
    return train_op_d, train_op_g


def placeholders():
    placeholder_input = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
    placeholder_latent = tf.placeholder(tf.float32, shape=[None, LATENT_DIM])
    return placeholder_input, placeholder_latent


def fill_feed_dictionary_for_discriminator(dataset, placeholder_input, placeholder_latent):
    input_next, _ = dataset.next_batch(BATCH_SIZE)
    latent_next = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, LATENT_DIM])
    return {
        placeholder_input: input_next,
        placeholder_latent: latent_next
    }


def fill_feed_dictionary_for_generator(placeholder_latent, nsamples=BATCH_SIZE):
    latent_next = np.random.uniform(-1.0, 1.0, size=[nsamples, LATENT_DIM])
    return {
        placeholder_latent: latent_next
    }


def visualize_samples(samples, step):
    fig = plt.figure(figsize=(5, 5))
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
        output_discriminator_dataset = inference_discriminator(placeholder_input)
        output_generator = inference_generator(placeholder_latent)
        output_discriminator_generated = inference_discriminator(output_generator, reuse=True)
        loss_exp_discriminator = loss_discriminator(output_discriminator_dataset, output_discriminator_generated)
        loss_exp_generator = loss_generator(output_discriminator_generated)
        train_op_discriminator, train_op_generator = train(loss_exp_discriminator, loss_exp_generator)

        # Create session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        tf.summary.FileWriter('./train', sess.graph).close()
        sess.run(init)

        print('%-20s | %-30s | %-30s' % ('Epoch', 'Loss(Discriminator)', 'Loss(Generator)'))
        print('-' * 86)
        for step in range(max_epochs):

            # Train the neural network graph
            start_time = time.time()
            _, loss_val_d = sess.run([train_op_discriminator, loss_exp_discriminator], feed_dict=fill_feed_dictionary_for_discriminator(dataset.train, placeholder_input, placeholder_latent))
            _, loss_val_g = sess.run([train_op_generator, loss_exp_generator], feed_dict=fill_feed_dictionary_for_generator(placeholder_latent))
            duration = time.time() - start_time

            # Visualize and report train stats
            if step % LOG_FREQUENCY == 0:
                print('%-20s | %-30s | %-30s' % ('%d' % (step), '%.5f' % (loss_val_d), '%.5f' % (loss_val_g)))
                samples = sess.run(output_generator, feed_dict=fill_feed_dictionary_for_generator(placeholder_latent, 25))
                visualize_samples(samples, step)


if __name__=='__main__':
    run_training(50001)