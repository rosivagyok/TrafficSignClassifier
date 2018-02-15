import data
import model
import preprocess
import visualize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator



# Load pickled data
train, test = data.load_traffic_sign_data('\\data\\train.p', '\\data\\test.p')
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Number of examples
n_train, n_test = X_train.shape[0], X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many classes?
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples  =", n_test)
print("Image data shape  =", image_shape)
print("Number of classes =", n_classes)


#first visual
visualize.show_classes(n_classes,n_train,n_test,X_train,X_test,y_train,y_test)

# preprocess train and test data
X_train_norm = preprocess.preprocess_features(X_train)
X_test_norm = preprocess.preprocess_features(X_test)

# split into train and validation
VAL_RATIO = 0.2
X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train, test_size=VAL_RATIO, random_state=0)


# create the generator to perform online data augmentation.
# Operations: Rotation (+-15 deg), random zoom for range (0.2f), horizontal and vertical shifts of samples
image_datagen = ImageDataGenerator(
    rotation_range=15.,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
    )

# take a random image from the training set
img_rgb = X_train[3]

# plot the original image
plt.figure(figsize=(1,1))
plt.imshow(img_rgb)
plt.title('Example of RGB image (class = {})'.format(y_train[0]))
plt.show()

# plot some randomly augmented images
rows, cols = 4, 10
fig, ax_array = plt.subplots(rows, cols)
for ax in ax_array.ravel():
    augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
    ax.imshow(np.uint8(np.squeeze(augmented_img)))

# hide labels in both axes
plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
plt.suptitle('Random examples of data augmentation (starting from the previous image)')
plt.show()



# placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)


# training pipeline
lr = 0.001
logits = model.my_net(x, n_classes=n_classes,keep_prob=keep_prob)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_function = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)

# metrics and functions for model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()


# training hyperparameters
BATCHSIZE = 128
EPOCHS = 30
BATCHES_PER_EPOCH = 5000


# start training
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):

        print("EPOCH {} ...".format(epoch + 1))

        batch_counter = 0
        # Takes numpy data & label arrays, and generates batches of augmented data
        for batch_x, batch_y in image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE):

            batch_counter += 1
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            if batch_counter % 200 == 0:
                print("BATCH {} / 5000".format(batch_counter), end='\r')
            if batch_counter == BATCHES_PER_EPOCH:
                break

        # at epoch end, evaluate accuracy on both training and validation set
        # you can restore a session with uncommenting the line below
        # checkpointer.restore(sess, '../checkpoints/traffic_sign_model.ckpt-27')
        train_accuracy = model.evaluate(X_train_norm, y_train,BATCHSIZE,accuracy_operation,x,y,keep_prob)
        val_accuracy = model.evaluate(X_val_norm, y_val,BATCHSIZE,accuracy_operation,x,y,keep_prob)
        print('Train Accuracy = {:.3f} - Validation Accuracy: {:.3f}'.format(train_accuracy, val_accuracy))
        
        # log current model weights at the end of each epoch
        checkpointer.save(sess, save_path='../checkpoints/traffic_sign_model.ckpt', global_step=epoch)


# evaluate the model on test data
with tf.Session() as sess:

    # load model weights for checkpoints (epoch with highest train/val accurancy)
    checkpointer.restore(sess, '../checkpoints/traffic_sign_model.ckpt-27')
    
    test_accuracy = evaluate(X_test_norm, y_test,BATCHSIZE,accuracy_operationx,x,y,keep_prob)
    print('Performance on test set: {:.3f}'.format(test_accuracy))