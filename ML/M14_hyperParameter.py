from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(inputs)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return {'batch_size':batches, 'optimizer':optimizers,
            'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters  = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)

search.fit(X_train, Y_train)

print(search.best_params_)