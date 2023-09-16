from util_fcn import *
from tensorflow.keras import backend as k
from keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
import time
from sklearn.metrics import confusion_matrix


params = DefineParameters(batch_train=1, epochs=1, batch_test=42, axis_select=0, mri_type=2, num_subject=(124,157,158,159,160,161,77,79,81,82,83,84))



#para entrenar con una sola sesion 
train_dataset = tf.data.Dataset.from_generator(lambda: just_one_ses(params),
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((None, 128, 128, 22), (None,)))

test_dataset = tf.data.Dataset.from_generator(lambda: test_data_generator(params),
                                              output_types=(tf.float32), 
                                              output_shapes=(None, 128, 128, 22))

# Generate a print
print('==========================================================================')
print(f'Starting',  'model training ... ...')

# build the model and compile
model = build_CNN_model(params)

# create callback
filepath = './Models/cnn/'  + '_resnet.epoch{epoch:02d}-loss{' \
                                                                       'val_loss:.2f}.hdf5 '
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]


# Fit data to model
inicio = time.time()
model.fit(train_dataset, epochs=params.epochs, steps_per_epoch=params.steps_per_epoch, validation_data=test_dataset,
          verbose=0, callbacks=callbacks)
fin = time.time()
print(fin-inicio)
#model.fit(train_dataset, epochs=params.epochs, steps_per_epoch=params.steps_per_epoch,validation_data=test_dataset,callbacks=[tqdm_callback], verbose=0)

# Generate prediction
print('==========================================================================')
print(f'Starting', params.axial_coords[params.axis_select], 'model testing ... ...')

prediction = model.predict(test_dataset, steps=1)


print(f'Results:')
print(prediction)
print(load_test_targets(params))
print('==========================================================================')

# Save model
model.save('./Models/cnn/cnn_' + params.axial_coords[params.axis_select], save_format='h5')

# Cleared tensorflow.keras session
k.clear_session()

print('FIN')
