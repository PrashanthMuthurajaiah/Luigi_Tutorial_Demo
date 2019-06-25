# Deep learning model training and inference pipeline
import luigi
import pandas as pd


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

    
