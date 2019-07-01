# Python Script for running specific tasks and Jobs of data pipeline

import configparser
import subprocess

config = configparser.ConfigParser()
configFilePath = './luigi_congig.cfg'
config.read(configFilePath)

task_to_be_executed = config['DEFAULT']['task_to_execute']
input_file_path = config['ModelBuildingandEvaluate']['data_extract_path_mb']


if task_to_be_executed == 'DownloadDataset':
    cmd = ['luigi', '--module', 'model_train_evaluate_pipeline', task_to_be_executed, '--data-path', input_file_path, '--local-scheduler']
    subprocess.run(cmd)

elif task_to_be_executed == 'DataPreprocessing':
    cmd = ['luigi', '--module', 'model_train_evaluate_pipeline', task_to_be_executed, '--data-path-extract', input_file_path, '--local-scheduler']
    subprocess.run(cmd)

else:
    cmd = ['luigi', '--module', 'model_train_evaluate_pipeline', task_to_be_executed, '--data-extract-path-mb', input_file_path, '--local-scheduler']
    subprocess.run(cmd)
