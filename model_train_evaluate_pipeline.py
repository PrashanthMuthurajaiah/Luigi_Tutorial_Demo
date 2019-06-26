# Deep learning model training and inference pipeline
import luigi
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


"""
Generally the first module or task in a data science pipeline will be data collection moduleself.
Data can be sourced/extracted from different sources
For this example, a already well prepared dataset is just read in to the application from its location
"""

class DownloadDataset(luigi.Task):

    data_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget('./base_dataframe.csv')

    def run(self):
        house_price_data = pd.read_csv(self.data_path)
        house_price_data.to_csv(self.output().path)
        return house_price_data

"""
This task/module will be handling the Preprocessing tasks that are necessary to perform before building the
the architecture of the model

The major difference in the DataPreprocessing Task will be that it will be overiding the requires() and will
obtain the necessary information from DownloadDataset class.
"""

class DataPreprocessing(luigi.Task):

    data_path_extract = luigi.Parameter()

    def requires(self):
        return DownloadDataset(self.data_path_extract)

    def output(self):
        #return luigi.LocalTarget(['./X_train.npy','X_val.npy'])
        return {'X_train_path':luigi.LocalTarget('./X_train.npy'),
        'X_val_path':luigi.LocalTarget('./X_val.npy'),
        'X_test_path':luigi.LocalTarget('./X_test.npy'),
        'Y_train_path':luigi.LocalTarget('./Y_train.npy'),
        'Y_val_path':luigi.LocalTarget('./Y_val.npy'),
        'Y_test_path':luigi.LocalTarget('./Y_test.npy')}


    def run(self):
        input_path = self.input().path
        base_dataframe = pd.read_csv(input_path)
        # Converting the dataframe into an array.
        dataset = base_dataframe.values
        X = dataset[:,0:10]
        Y = dataset[:,10]

        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)

        # Using sklearn test_train_split to split data into test and train dataset
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

        # Usinf train_test_split to create validation data
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

        # Printing the shape of all the variables
        print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

        # Saving the numpy arrays to their designated location which will be utilized while model creation
        np.save('./X_train.npy', X_train)
        np.save('./X_val.npy', X_val)
        np.save('./X_test.npy', X_test)
        np.save('./Y_train.npy', Y_train)
        np.save('./Y_val.npy', Y_val)
        np.save('./Y_test.npy', Y_test)


"""
This Module/Task is responsible for building model architecture.
"""

class ModelBuilding(luigi.Task):
    data_extract_path_mb = luigi.Parameter()

    def requires(self):
        return DataPreprocessing(self.data_extract_path_mb)

    def output(self):
        return luigi.LocalTarget('./Model/house_price_predict.h5')

    def run(self):
        numpy_array_data_path = self.input()
        X_train_saved_path = numpy_array_data_path['X_train_path'].path
        X_val_saved_path = numpy_array_data_path['X_val_path'].path
        X_test_saved_path = numpy_array_data_path['X_test_path'].path
        Y_train_saved_path = numpy_array_data_path['Y_train_path'].path
        Y_val_saved_path = numpy_array_data_path['Y_val_path'].path
        Y_test_saved_path = numpy_array_data_path['Y_test_path'].path

        #Loading Numpy arrays
        X_train_v2 = np.load(X_train_saved_path)
        X_val_v2 = np.load(X_val_saved_path)
        X_test_v2 = np.load(X_test_saved_path)
        Y_train_v2 = np.load(Y_train_saved_path)
        Y_val_v2 = np.load(Y_val_saved_path)
        Y_test_v2 = np.load(Y_test_saved_path)

        # Define the architecture of the model
        model = Sequential([Dense(32, activation='relu', input_shape=(10,)), Dense(32, activation='relu'),  Dense(1, activation='sigmoid'),])
        model.compile(optimizer='sgd',loss='binary_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train_v2, Y_train_v2,batch_size=32, epochs=100,validation_data=(X_val_v2, Y_val_v2))
        model.save('./Model/house_price_predict.h5')

# """
# Module/Tasks which performs the training.
# """
#
# class ModelTraining(luigi.Task):
#     data_extract_path_mt = luigi.Parameter()
#
#     def requires(self):
#         return ModelBuilding(self.data_extract_path_mt)
#
#     def output(self):
#         return luigi.LocalTarget('./Models/Model_with_weights.h5')
#
#     def run(self):
#         model_path = self.input().path
#         new_model = load_model(model_path)
