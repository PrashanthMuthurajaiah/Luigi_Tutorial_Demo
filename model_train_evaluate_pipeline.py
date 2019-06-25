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

    def run(self):
        house_price_data = pd.read_csv(self.data_path)
        print(house_price_data)
