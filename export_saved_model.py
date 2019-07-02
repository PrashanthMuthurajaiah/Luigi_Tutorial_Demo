import tensorflow as tf
from keras.models import load_model

# export_path contains the location of the model and the name
tf.keras.backend.set_learning_phase(0) #Ignore the drop out at inference


export_path = './house_price_predictor/1'

#Initiate the keras session and save the models
with tf.keras.backend.get_session() as sess:
    model = load_model('./Model/house_price_predict.h5')
    tf.saved_model.simple_save(
    sess,
    export_path,
    inputs = {'input_shape': model.input},
    outputs = {t.name: t for t in model.outputs})
