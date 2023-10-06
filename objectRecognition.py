import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from matplotlib import pyplot as plt
import datetime
from tensorboard.plugins.hparams import api as hp
from keras.regularizers import l2
from keras.utils import to_categorical

## settings --------------------------------------------
log_dir='logs\\fit\\'+datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
plotFig=0
lr=0.001
momentum=0.04
epoch=50
batchsize=128
metric=['accuracy']
valid_sample_num=5000

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=lr)

relu = keras.layers.ReLU()
BatchNormal = keras.layers.BatchNormalization()
maxpool = keras.layers.MaxPool2D((2,2))
GlobalAvgPool = keras.layers.GlobalAveragePooling2D()
flatten = keras.layers.Flatten()
input = keras.Input(shape=(32,32,3))
out_dense = keras.layers.Dense(10)

## image preprocessing ---------------------------------
def scale(image):
    image = tf.cast(image, tf.float32)
    image/=255.

    return image

## load and plot dataset -------------------------------
def load_data():
    (x_train_validation, y_train_validation) , (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train_validation=scale(x_train_validation)
    x_test=scale(x_test)

    if(plotFig):
        plt.figure(figsize=(10,10))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_train_validation[i], cmap=plt.cm.binary)
        plt.show()

    n_samples=x_train_validation.shape[0]
    x_train=x_train_validation[:n_samples-valid_sample_num,:,:,:]
    x_valid=x_train_validation[-valid_sample_num:,:,:,:]
    y_train=y_train_validation[:n_samples-valid_sample_num,:]
    y_valid=y_train_validation[-valid_sample_num:,:]

    return (x_train,y_train,x_valid,y_valid,x_test,y_test)

def plot_res(history):
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.title('Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='valid')
    plt.subplot(2,1,2)
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='valid')
    plt.show()

## CNN model config -------------------------------------
def cnn_model():
    conv1 = keras.layers.Conv2D(32, 5)
    conv2 = keras.layers.Conv2D(64, 3)
    conv3 = keras.layers.Conv2D(128, 3)
    flatten = keras.layers.Flatten()
    dropout = keras.layers.Dropout(0.2)
    dense1 = keras.layers.Dense(1024)

    x = maxpool(relu(conv1(input)))
    x = maxpool(relu(conv2(x)))
    x = maxpool(relu(conv3(x)))
    x = flatten(x)
    x = relu(dense1(x))
    output = out_dense(x)

    model=keras.Model(inputs=input, outputs=output, name='functional_model')
    print(model.summary())

    return model

def VGG_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer=l2(0.001), input_shape=(32,32,3)),
        keras.layers.Conv2D(32, 3, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer='he_uniform', padding='same'),
        keras.layers.Conv2D(64, 3, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(128, 3, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer='he_uniform', padding='same'),
        keras.layers.Conv2D(128, 3, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer='he_uniform', padding='same'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    print(model.summary())
    return model
    

def inception_module(layer_in):
    conv1 = keras.layers.Conv2D(64, 1, padding='same', activation='relu')(layer_in)
    conv3 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(layer_in)
    conv5 = keras.layers.Conv2D(32, 5, padding='same', activation='relu')(layer_in)
    pool = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out

def googleNet():
    dense1 = keras.layers.Dense(512)

    x = inception_module(input)
    x = inception_module(x)
    x = flatten(x)
    x = relu(dense1(x))
    output = out_dense(x)
    model=keras.Model(inputs=input, outputs=output, name='functional_model')
    print(model.summary())

    return model

def residual_module(layer_in, n_filters):
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = keras.layers.Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        
    conv1 = keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    conv2 = keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    layer_out = keras.layers.add([conv2, merge_input])
    layer_out = keras.layers.Activation('relu')(layer_out)
    return layer_out


def resNet():
    x = residual_module(input, 16)
    dense1 = keras.layers.Dense(512)
    x = flatten(x)
    x = relu(dense1(x))
    output = out_dense(x)
    model=keras.Model(inputs=input, outputs=output, name='functional_model')
    print(model.summary())
    return model

## build model --------------------------------------------
def build_model(model, x_train, y_train, x_valid, y_valid):
    datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, 
    horizontal_flip=True,featurewise_center=True,featurewise_std_normalization=True,rotation_range=20)

    model.compile(optimizer=optim, loss=loss, metrics=metric)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                        min_delta=0, patience=5, verbose=0, restore_best_weights=True)

    tensorboard_callback=keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize, validation_data=(x_valid,y_valid),
        callbacks=[tensorboard_callback, early_stopping], verbose=2)

    return (model, history)
    
    



(x_train,y_train,x_valid,y_valid,x_test,y_test)=load_data()

# model = cnn_model()
# model = VGG_model()
# model = googleNet()
model = resNet()
(model, history)=build_model(model, x_train, y_train, x_valid, y_valid)
model.evaluate(x_test, y_test, batch_size=batchsize, verbose=2)

plot_res(history.history)
keras.utils.plot_model(model,to_file=log_dir+"model.png")