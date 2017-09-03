import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#process settings
process_video = True # create a video
process_train = False  # train a network
process_load = True  # load a network
process_save = False   # save a network
#global variables for process image function
image_shape = (1,1)
sess = tf.Session()
keep_prob = tf.placeholder(tf.float32)
loggits = tf.placeholder(tf.int32, [None, None, None, 2])
input_image = tf.placeholder(tf.int32, [None, None, 3])


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    kp = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, kp, w3, w4, w7


#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output size 4096
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output size 512
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output size 256
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # convert output layers to 1x1 convolutional layers

    conv_1x1_lay7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1_lay4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.layers.conv2d_transpose(conv_1x1_lay7, num_classes, 4, 2, 'same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  # scale up by x2
    output = tf.add(output, conv_1x1_lay4)  # first skip layer

    conv_1x1_lay3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, 'same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  # scale up by x2
    output = tf.add(output, conv_1x1_lay3)  # second skip layer

    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, 'same',kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))  # scale up by x8 to get original image size

    return output


#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss


#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
            feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-4}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            print('Epoch ', epoch, ' Batch ', batch, ' Loss ', loss, flush=True)

    pass


#tests.test_train_nn(train_nn)

def process_image(image):
    
    image = scipy.misc.imresize(image, image_shape)
    
    im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: [image]})
        

    label_index = np.argmax(im_softmax, axis=2)

    value_fill_1 = label_index.copy()
    value_fill_1.fill(1)
    value_fill_2 = label_index.copy()
    value_fill_2.fill(2)
    value_fill_3 = label_index.copy()
    value_fill_3.fill(3)

    segmentation1 = np.equal(label_index, value_fill_1)
    segmentation2 = np.equal(label_index, value_fill_2)
    segmentation3 = np.equal(label_index, value_fill_3)
    
    segmentation1 = segmentation1.reshape(image_shape[0], image_shape[1], 1)
    segmentation2 = segmentation2.reshape(image_shape[0], image_shape[1], 1)
    segmentation3 = segmentation3.reshape(image_shape[0], image_shape[1], 1)
    mask1 = np.dot(segmentation1, np.array([[0, 255, 0, 127]]))
    mask1 = scipy.misc.toimage(mask1, mode="RGBA")
    mask2 = np.dot(segmentation2, np.array([[0, 0, 255, 127]]))
    mask2 = scipy.misc.toimage(mask2, mode="RGBA")
    mask3 = np.dot(segmentation3, np.array([[255, 242, 0, 127]]))
    mask3 = scipy.misc.toimage(mask3, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask1, box=None, mask=mask1)
    street_im.paste(mask2, box=None, mask=mask2)
    street_im.paste(mask3, box=None, mask=mask3)
    
    return np.array(street_im)
   
   
def run():
    global image_shape, sess, logits, keep_prob, input_image
    num_classes = 4
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pre-trained vgg model
    #helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        epochs = 30
        batch_size = 8
        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, optimizer, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if(process_load):
            checkpoint = tf.train.get_checkpoint_state("save_network")
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")

        if(process_train):
            train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, input_image,
                     correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        if(process_video):
            Output_video = 'street_drive_sw_tracked.mp4'
            Input_video = 'street_drive.mp4'

            video_output = Output_video
            clip1 = VideoFileClip(Input_video)
            video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
            video_clip.write_videofile(video_output, audio=False)

        if(process_save):
            save_path = saver.save(sess, "./save_network/model_60_8")

def my_test():
    x = tf.placeholder(tf.float32,[None,None,None,2])
    y = tf.layers.conv2d_transpose(x,2,200)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = tf.trainable_variables()

        pass
        out = sess.run(x[0])
        pass




if __name__ == '__main__':
    run()
