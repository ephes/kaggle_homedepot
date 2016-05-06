import numpy as np
import xgboost as xgb

from scipy.sparse import hstack

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.layers.recurrent import LSTM


class ClippedMixin:
    def predict(self, X):
        predictions = super().predict(X)
        return np.clip(predictions, 1, 3)


class CachedMixin:
    trained = set()
    cache = {}

    def fit(self, X_train, y):
        train_hash = hash(str(X_train))
        if train_hash not in self.trained:
            fitted = super().fit(X_train, y)
            self.trained.add(train_hash)

    def predict(self, X_test):
        test_hash = hash(str(X_test))
        if test_hash not in self.cache:
            predictions = super().predict(X_test)
            self.cache[test_hash] = predictions
        return self.cache[test_hash]


class ClippedNN(ClippedMixin, CachedMixin):
    trained = set()
    cache = {}

    def __init__(self, data_dim):
        self.scaler = StandardScaler(with_mean=False)

        self.model = Sequential()
        self.model.add(Dense(input_dim=data_dim, output_dim=32, init='glorot_uniform'))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=32))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=32, init='glorot_uniform'))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=32))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=32))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=32))
        self.model.add(Activation('prelu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(input_dim=32, output_dim=1))
        #self.model.compile(loss='mean_absolute_error', optimizer='rmsprop')
        self.model.compile(loss='mean_absolute_error', optimizer='adam')


#        benchmark
#        self.model = Sequential()
#        self.model.add(Dense(input_dim=data_dim, output_dim=32))
#        self.model.add(Activation('tanh'))
#        #self.model.add(Dropout(0.1))
#        #self.model.add(Dense(input_dim=32, output_dim=32))
#        #self.model.add(Activation('tanh'))
#        #self.model.add(Dropout(0.1))
#        #self.model.add(Dense(input_dim=32, output_dim=32))
#        #self.model.add(Activation('tanh'))
#        #self.model.add(Dropout(0.1))
#        self.model.add(Dense(input_dim=32, output_dim=1))
#        self.model.compile(loss='mean_absolute_error', optimizer='rmsprop')
#        #self.model.compile(loss='mean_absolute_error', optimizer='adam')


#        self.model = Sequential([
#            Dense(64, input_dim=584),
#            Activation('relu'),
#            Activation('softmax'),
#        ])
#        self.model.add(Dense(64, 1))
#
#        #self.model.add(Dropout(0.5))
#        #self.model.add(Dense(1000, 1000, init='glorot_uniform'))
#        self.model.add(Activation('linear'))
#        self.model.compile(loss='mean_squared_error', optimizer="adam")

    def fit(self, dense, svd, sparse, y):
        X_train = np.hstack((dense, svd))
        train_hash = hash(str(X_train))
        if train_hash not in self.trained:
            X_scaled = self.scaler.fit_transform(X_train)
            X_scaled = normalize(X_scaled)
            print(X_train.shape)
            #X_train = hstack((X_train, sparse))
            fitted = self.model.fit(X_scaled, y, batch_size=400, nb_epoch=10,
                                    validation_split=0.05)
            self.trained.add(train_hash)

    def predict(self, dense, svd, sparse):
        X_test = np.hstack((dense, svd))
        test_hash = hash(str(X_test))
        if test_hash not in self.cache:
            X_scaled = self.scaler.fit_transform(X_test)
            X_scaled = normalize(X_scaled)
            #X_test = hstack((X_test, sparse))
            y_pred = self.model.predict(X_scaled)
            self.cache[test_hash] = y_pred
        print(self.cache[test_hash].shape)
        return self.cache[test_hash]


class ClippedRF(ClippedMixin, RandomForestRegressor):
    trained = set()
    cache = {}

    def fit(self, dense, svd, sparse, y):
        #X_train = np.hstack((dense, svd))
        X_train = dense
        #X_train = hstack((X_train, sparse))
        train_hash = hash(str(X_train))
        if train_hash not in self.trained:
            fitted = super().fit(X_train, y)
            self.trained.add(train_hash)
        return self

    def predict(self, dense, svd, sparse):
        #X_test = np.hstack((dense, svd))
        X_test = dense
        #X_test = hstack((X_test, sparse))
        test_hash = hash(str(X_test))
        if test_hash not in self.cache:
            predictions = super().predict(X_test)
            self.cache[test_hash] = predictions
        return self.cache[test_hash]


class ClippedSVR(ClippedMixin, CachedMixin, SVR):
    def fit(self, dense, svd, sparse, y):
        X_train = np.hstack((dense, svd))
        #X_train = hstack((X_train, sparse))
        return super().fit(X_train, y)

    def predict(self, dense, svd, sparse):
        X_test = np.hstack((dense, svd))
        #X_test = hstack((X_test, sparse))
        return super().predict(X_test)


class ClippedETRStacked(ClippedMixin, ExtraTreesRegressor):
    pass


class ClippedETR(ClippedMixin, ExtraTreesRegressor):
    trained = set()
    cache = {}

    def fit(self, dense, svd, sparse, y):
        X_train = np.hstack((dense, svd))
        #X_train = hstack((X_train, sparse))
        train_hash = hash(str(X_train))
        if train_hash not in self.trained:
            fitted = super().fit(X_train, y)
            self.trained.add(train_hash)
        return self

    def predict(self, dense, svd, sparse):
        X_test = np.hstack((dense, svd))
        #X_test = hstack((X_test, sparse))
        test_hash = hash(str(X_test))
        if test_hash not in self.cache:
            predictions = super().predict(X_test)
            self.cache[test_hash] = predictions
        return self.cache[test_hash]



class ClippedLR(ClippedMixin, LinearRegression):
    def fit(self, X_train, y):
        super().fit(scale(X_train, with_mean=False), y)

    def predict(self, X_test):
        super().predict(scale(X_test, with_mean=False))


class ClippedXGBStacked(ClippedMixin, XGBRegressor):
    pass


class ClippedRFRStacked(ClippedMixin, RandomForestRegressor):
    pass


class ClippedXGB(ClippedMixin, CachedMixin, XGBRegressor):
    def fit(self, dense, svd, sparse, y):
        X_train = np.hstack((dense, svd))
        X_train = hstack((X_train, sparse))
        return super().fit(X_train, y)

    def predict(self, dense, svd, sparse):
        X_test = np.hstack((dense, svd))
        X_test = hstack((X_test, sparse))
        return super().predict(X_test)


class LinearXGB(ClippedMixin):
    trained = set()
    cache = {}

    def __init__(self, params, num_rounds):
        self.params = params
        self.scaler = StandardScaler(with_mean=False)
        self.num_rounds = num_rounds

    def fit(self, dense, svd, sparse, y):
        X_train = np.hstack((dense, svd))
        #X_train = hstack((X_train, sparse))
        train_hash = hash(str(X_train))
        if train_hash not in self.trained:
            X_scaled = self.scaler.fit_transform(X_train)
            X_scaled = normalize(X_scaled)
            dtrain = xgb.DMatrix(X_scaled, label=y)
            watchlist = [(dtrain, 'train')]
            self.bst = xgb.train(self.params, dtrain, self.num_rounds)#, watchlist)
            self.trained.add(train_hash)

    def predict(self, dense, svd, sparse):
        X_test = np.hstack((dense, svd))
        #X_test = hstack((X_test, sparse))
        test_hash = hash(str(X_test))
        if test_hash not in self.cache:
            #X_scaled = X_test
            X_scaled = self.scaler.fit_transform(X_test)
            X_scaled = normalize(X_scaled)
            dtest = xgb.DMatrix(X_scaled)
            #dtest = xgb.DMatrix(X_test)
            y_pred = self.bst.predict(dtest)
            self.cache[test_hash] = y_pred
        return self.cache[test_hash]


class Ensemble:
    def __init__(self, clfs):
        self.clfs = clfs
        self.sum_weights = float(sum([w for c, w in self.clfs]))
        
    def fit(self, dense, svd, sparse, y):
        for clf, weight in self.clfs:
            clf.fit(dense, svd, sparse, y)
    
    def predict_raw(self, dense, svd, sparse):
        results = []
        for clf, weight in self.clfs:
            prediction = clf.predict(dense, svd, sparse)
            prediction = prediction.reshape(-1)
            results.append(prediction)
        return np.vstack(results).T

    def predict(self, dense, svd, sparse):
        result = np.zeros(dense.shape[0])
        for clf, weight in self.clfs:
            print(weight)
            prediction = clf.predict(dense, svd, sparse)
            prediction = prediction.reshape(-1)
            result += weight * prediction
        return result / self.sum_weights
