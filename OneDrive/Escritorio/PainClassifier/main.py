from data_generator import *
from classifiers_models import *
from metrics import *
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.vgg16 import preprocess_input


subjects = [124,157,158,159,140,76,77,79,81,82,83,84]
sessions = [2,3]
params = DefineParameters(model="VGG19", batch_train=1, epochs=10, batch_test=1, fMRI_Modality='rest', subjects=subjects,
                          sessions=sessions)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

scores, histories = list(), list()
for train_ix, test_ix in kfold.split(params.filenames, params.label_names):   
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data_generator_short(params, train_ix),
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=((None, 128, 128, 22), (None, 2)))
    
    test_dataset = tf.data.Dataset.from_generator(lambda: test_data_generator_short(params, test_ix),
                                                   output_types=(tf.float32),
                                                   output_shapes=((None, 128, 128, 22)))
    
    # build the model and compile
    if params.model == "VGG19":
        model = vgg19()
    elif params.model == "VGG16":
        model = vgg16()
    elif params.model == "2D":
        build_CNN_model2D
    else:
        print("Ningun modelo encontrado")
        

    # Fit data to model
    history = model.fit(train_dataset, epochs=params.epochs, steps_per_epoch=620*len(train_ix), verbose=1)
    
    test_label = test_labels(params, test_ix)

    ROC(params,model,test_dataset,test_label)
    
    save_performance(params)

    
    _,acc = model.evaluate(test_dataset, verbose=1, steps=len(test_ix))
    scores.append(acc)
    histories.append(history)

 
summarize_diagnostics(histories)
summarize_performance(scores)

  
print('end')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


