import numpy as np
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.callbacks import Callback
from keras.metrics import SparseCategoricalCrossentropy
from matplotlib import pyplot as plt
from matplotlib.colors import CSS4_COLORS
import time


def load_digits():
    digits = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = split(digits[0][0], digits[0][1], test_size=10000)
    sets = ((x_train, y_train), (x_val, y_val), digits[1])
    train_n, val_n, test_size = sets[0][0].shape[0], sets[1][0].shape[0], sets[2][0].shape
    msg = 'Loading {} training samples, {} validation samples, and {} test samples of size {} by {}'
    print(msg.format(train_n, val_n, *test_size))
    print('from the MNIST handwritten digits database')
    shape = (*test_size[1:], 1)
    return {key: {'x': np.expand_dims(x, 3).astype(np.float32) / 255., 'y': y}
            for key, (x, y) in zip(('training', 'validation', 'testing'), sets)}, shape


def network(shape, dropout=None):
    net = Sequential()
    net.add(Conv2D(28, kernel_size=3, strides=2, input_shape=shape))
    net.add(Flatten())
    net.add(Dense(128, activation=tf.nn.relu))
    if dropout is not None:
        net.add(Dropout(dropout))
    net.add(Dense(10, activation=tf.nn.softmax))
    return net


def count_weights(net):
    return sum([np.prod(w.shape) for w in net.trainable_weights])


class StoreStatistics(Callback):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset
        self.risks = {}

    def get_stats(self):
        return self.risks

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        for which in ('training', 'validation'):
            dataset = self.data[which]
            x, y = dataset['x'], dataset['y']
            perf = self.model.evaluate(x, y, verbose=0)
            self.risks[which] = {'risk': [perf[0]], 'accuracy': [perf[1]]}

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print(epoch, end=' ')
        for which in ('training', 'validation'):
            dataset = self.data[which]
            x, y = dataset['x'], dataset['y']
            perf = self.model.evaluate(x, y, verbose=0)
            self.risks[which]['risk'].append(perf[0])
            self.risks[which]['accuracy'].append(perf[1])

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        print()


def fit(dataset, epochs, batch=None, opt='SGD', verb=0):
    model.load_weights('model.h5')
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if batch is None:
        batch = len(dataset['training']['y'])
    store_stats = StoreStatistics(dataset)
    print('Training')
    start_time = time.time()
    h = model.fit(x=dataset['training']['x'], y=dataset['training']['y'],
                  validation_data=(dataset['validation']['x'], dataset['validation']['y']),
                  batch_size=batch, epochs=epochs, verbose=verb,
                  callbacks=[store_stats])
    end_time = time.time()
    print('Done training. ', end='')
    risks = store_stats.get_stats()
    hist = {'risk': {'training': risks['training']['risk'],
                     'validation': risks['validation']['risk']},
            'accuracy': {'training': risks['training']['accuracy'],
                         'validation': risks['validation']['accuracy']}}
    return hist, end_time - start_time


def plot_history(hist, h, batch, opt, dataset, t):
    n_epochs = len(hist['risk']['training'])
    epochs = list(range(n_epochs))

    print('Evaluating final model', end='')
    acc = {name: 100. * h.evaluate(samples['x'], samples['y'], verbose=0)[1]
           for name, samples in dataset.items()}
    print('. Done.')

    fs, ms, lw = 18, 4, 2
    plt.figure(figsize=(15, 5))
    for k, (title, quantity) in enumerate(hist.items()):
        plt.subplot(1, 2, k + 1)
        for set_type, values in quantity.items():
            plt.plot(epochs, values, lw=lw, marker='.', markersize=3*ms, label=set_type)
        plt.legend(fontsize=fs)
        if len(epochs) < 20:
            plt.xticks(epochs)
        else:
            plt.xticks(epochs[::10])
        plt.xlabel('epoch', fontsize=fs)
        plt.ylabel(title, fontsize=fs)

    if batch is None:
        batch = len(dataset['training']['y'])
    fmt = 'Batch size {}, {} optimizer. ' + \
        'Final accuracies: training {:.1f}%, validation {:.1f}%, testing {:.1f}% ({} seconds)'
    plt.gcf().suptitle(fmt.format(batch, opt, acc['training'], acc['validation'],
                                  acc['testing'], np.round(t).astype(int)), fontsize=16)
    figure_name = 'b_{}_o_{}.png'.format(batch, opt)
    plt.savefig(figure_name)
    print('Plots saved to {}'.format(figure_name))
    plt.show()


verbose = 0

data, input_shape = load_digits()
model = network(input_shape, dropout=None)
model.save_weights('model.h5')
print('The model has {} trainable weights'.format(count_weights(model)))

optimizer = 'SGD'
for batch_size in (100, 1000, None):
    print('Batch size {}, {} optimizer'.format(batch_size, optimizer))
    epochs = 50 if batch_size is None else 10
    history, duration = fit(data, epochs=epochs, batch=batch_size,
                            opt=optimizer, verb=verbose)
    plot_history(history, model, batch_size, optimizer, data, duration)

batch_size, optimizer, epochs = 100, 'Adam', 10
print('Batch size {}, {} optimizer'.format(batch_size, optimizer))
history, duration = fit(data, epochs=epochs, batch=batch_size,
                        opt=optimizer, verb=verbose)
plot_history(history, model, batch_size, optimizer, data, duration)
