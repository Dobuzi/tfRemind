import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.3:
            print('\nStop training...')
            self.model.stop_training = True
