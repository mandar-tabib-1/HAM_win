# System
import copy
import sys
import time,datetime
from datetime import datetime
import os
import gc
import pickle
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# Mathematical operations / data manipulations
import numpy as np
import pandas as pd
import scipy as sc

# Graphics
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})
import matplotlib.animation as manimation
import matplotlib as mpl
import matplotlib.dates as mdates


# Deep learning / ML part
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.layers import Dropout,RepeatVector
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.losses import mse


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#For loading and saving data using dump function of joblib.
import joblib

# for hyperparameter tuning
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour

################################################################
# **** Functi   ons for use *****

#Optuna based optimization
best_booster = None
gbm = None

def objective(trial,xtrain,ytrain):# For LSTM model
    global gbm
         
    Q=xtrain.shape[1] # -1 was used because featuer of Reynolds number was added.
    
    filenamelstm='LSTM Models 02_02_2024'
    
    if os.path.isdir("./"+filenamelstm):
        print('LSTM models folder already exists')
    else: 
        print('Creating LSTM models folder')
        os.makedirs("./"+filenamelstm)
    
    # Removing old models
    model_name = filenamelstm + '/ANN-Corrector.h5'
    if os.path.isfile(model_name):
        os.remove(model_name)

    ## Shuffling data    
    #perm = np.random.permutation(m)
    #xtrain = xtrain[perm,:,:]
    #ytrain = ytrain[perm,:]
    
    lstmtype=2
    
    lookback=3

    epochs=200
    
    batch_size=10
        
    model3 = create_model_ANN(trial,xtrain,ytrain)
    
    #Criteria for early stoping, model saving and reducing learning rate on the fly.
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001)
    callbacks=[TFKerasPruningCallback(trial, "val_loss"),es,mc,reduce_lr]
    print(epochs,batch_size,'epochs')   
    
    # run the model
    history = model3.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,validation_split=0.2,\
                         callbacks=[TFKerasPruningCallback(trial, "val_loss")]) #inculde ,es,mc,reduce_lr

    # evaluate the model
    scores = model3.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model3.metrics_names[1], scores[1]*100))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
        
    return val_loss[-1]

###############################################################################################################33


# Wrapper function to include additional arguments
def wrapped_objective(trial):
    return objective(trial, X_train, y_train)

def create_model(trial,xtrain, ytrain):
    # We optimize the numbers of layers, their units and weight decay parameter.

    n_layers = trial.suggest_int("n_layers", 4, 6)
    
    #weight_decay = trial.suggest_float("weight_decay",  2e-6, 1e-1, log=True)
    
    #lr = trial.suggest_float("lr", 0.05, 0.06, log=True)
      
    #momentum = trial.suggest_float("momentum", 0.8, 1.0)
    
    model = Sequential()
    
    

    for i in range(n_layers):    #create model over n layers
        
        dropout = 0. # trial.suggest_float("dropout_l{}".format(i), 0.0, 0.42)

        num_hidden = trial.suggest_int("n_units_l{}".format(i), 20, 50, log=False)         
        
        if i < max(range(n_layers)):
            rs = True
        else:
            rs = False # final hidden 

        Q=ytrain.shape[0]
        
        #print("checking shape", xtrain.shape[1:])

        model.add(LSTM(num_hidden, input_shape=(xtrain.shape[1:]), return_sequences=rs))

        # model.add(Dropout(rate=dropout)) #Uncomment to include dropout 

        #model.add(BatchNormalization())                
        
    # Add last linear dense layer for regression with linear activation
    model.add(Dense(Q, activation='linear')) #kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
     
    #Compile model for regression using mse loss and adam optimizer.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    
    # Compile model for classification by using appropriate loss and meterics
    #model.compile(
    #    optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
    #    loss="sparse_categorical_crossentropy",
    #    metrics=["accuracy"],
    #)

    return model

###############################################################################################333333
def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


############################################################################################################3333
      
#Reuse the best optuna model
def create_model1(dict_para,xtrain,ytrain):
    
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.utils import get_custom_objects

    # Define the custom sinusoidal activation function
    def sinusoidal(x):
        return tf.math.sin(x)

# Register the custom activation function
    get_custom_objects().update({'sinusoidal': Activation(sinusoidal)})
    
    n_layers = dict_para['n_layers']
    
    #from tensorflow.keras.models import Sequential
    model = Sequential()

    for i in range(n_layers):
        
        # dropout = dict_para['dropout_l0'] #trial.suggest_float("dropout_l{}".format(i), 0.21, 0.42)

        num_hidden = dict_para['n_units_l'+str(i)] #dict_para['n_units_l0'] #trial.suggest_int("n_units_l{}".format(i), 20, 150, log=True)
        
        activation2=dict_para['activation2_'+str(i)]  
                
        print("num_hidden in layer ",i,"  ",num_hidden, activation2)
        
        if i < max(range(n_layers)):
            rs = True
        else:
            rs = False # final hidden 

        Q=1 #ytrain.shape[0]

        if i==0: 
            model.add(Dense(num_hidden, activation=activation2,input_shape=(xtrain.shape[1],))) #, return_sequences=rs for lstm.
        else:
            model.add(Dense(num_hidden, activation=activation2)) #, return_sequences=rs

        # model.add(Dropout(rate=dropout)) 

    model.add(Dense(Q, activation='linear')) 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35) #keras.callbacks.
    mc = ModelCheckpoint(filepath=os.getcwd(),filename='{epoch}-{val_loss:.5f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
    callbacks=[es,mc,reduce_lr]
    
    epochs=200
    
    batch_size=10   
    
    # retrain using the best model
    history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[es,mc,reduce_lr]) #callbacks=[es,mc,reduce_lr]

    # evaluate the model on train and test data
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return model,loss,val_loss


def create_model_ANN(trial,xtrain, ytrain):
    # We optimize the numbers of layers, their units and weight decay parameter.
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.utils import get_custom_objects

    # Define the custom sinusoidal activation function
    @tf.keras.utils.register_keras_serializable(package="my_package", name="sinusoidal")
    def sinusoidal(x):
        return tf.math.sin(x)

# Register the custom activation function
    get_custom_objects().update({'sinusoidal': sinusoidal})
    n_layers = trial.suggest_int("n_layers", 2, 5)
    
    #weight_decay = trial.suggest_float("weight_decay",  2e-6, 1e-1, log=True)
    
    #lr = trial.suggest_float("lr", 0.05, 0.06, log=True)
      
    #momentum = trial.suggest_float("momentum", 0.8, 1.0)
    
    model = Sequential()
    
    for i in range(n_layers):    #create model over n layers
        
        dropout = 0. # trial.suggest_float("dropout_l{}".format(i), 0.0, 0.42)

        num_hidden = trial.suggest_int("n_units_l{}".format(i), 20, 30, log=False)    
        
        activation2 = trial.suggest_categorical('activation2_{}'.format(i), ['sinusoidal', 'relu']) #'relu', 'leaky_relu',     'tanh'
        
        if i < max(range(n_layers)):
            rs = True
        else:
            rs = False # final hidden 

        Q=1 #ytrain.shape[0]
        
            
        if i==0: 
            model.add(Dense(num_hidden, activation=activation2,input_shape=(xtrain.shape[1],))) #, return_sequences=rs for lstm.
        else:
            model.add(Dense(num_hidden, activation=activation2)) #, return_sequences=rs
            

        # model.add(Dropout(rate=dropout)) #Uncomment to include dropout 

        #model.add(BatchNormalization())                
        
    # Add last linear dense layer for regression with linear activation
    model.add(Dense(Q, activation='linear')) #kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
     
    #Compile model for regression using mse loss and adam optimizer.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    
    # Compile model for classification by using appropriate loss and meterics
    #model.compile(
    #    optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
    #    loss="sparse_categorical_crossentropy",
    #    metrics=["accuracy"],
    #)

    return model

###############################################################################################333333

def compile_model_and_save_from_dict(dict_para,Q,inpshape): # Q=ytrain.shape[1],inpshape=xtrain.shape[1:]
    
    n_layers = dict_para['n_layers']
    
    from tensorflow.keras.models import Sequential
    model = Sequential()

    for i in range(n_layers):
        
        dropout = dict_para['dropout_l0'] #trial.suggest_float("dropout_l{}".format(i), 0.21, 0.42)

        num_hidden = dict_para['n_units_l0'] #trial.suggest_int("n_units_l{}".format(i), 20, 150, log=True)
        
        if i < max(range(n_layers)):
            rs = True
        else:
            rs = False # final hidden 

        #Q=ytrain.shape[1]

        model.add(LSTM(num_hidden, input_shape=(inpshape), return_sequences=rs))

        model.add(Dropout(rate=dropout)) 

    model.add(Dense(Q, activation='linear')) 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) #keras.callbacks.
    mc = ModelCheckpoint(filepath=os.getcwd(),filename='{epoch}-{val_loss:.5f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
    callbacks=[es,mc,reduce_lr]
    
    epochs=200
    
    batch_size=10   
        
    return model,callbacks,epochs,batch_size

# Important : check nfrac below
def split_timeseries(df,nfrac=0.8):
    nlen=df.shape[0]
    nrow=int(nfrac*nlen)
    train_dataset = df.iloc[:nrow, :]
    test_dataset = df.iloc[nrow:,:]
    return train_dataset,test_dataset

def split_timeseries_3D(df,nfrac=0.8):
    nlen=df.shape[0]
    nrow=int(nfrac*nlen)
    train_dataset = df[:nrow]
    test_dataset = df[nrow:]
    return train_dataset,test_dataset

def split(df):
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    return train_dataset,test_dataset
    
def scale_ANN(train_features,test_features,filename_4_scalingfunction):
    normalizer = MinMaxScaler(feature_range=(-1,1))
    print(type(train_features))
    if isinstance(train_features,pd.core.series.Series):
        normalizer_fit = normalizer.fit(train_features.values.reshape(-1,1))
        norm_train_features=normalizer_fit.transform(train_features.values.reshape(-1,1))
        norm_test_features=normalizer_fit.transform(test_features.values.reshape(-1,1))
    else:
        normalizer_fit = normalizer.fit(train_features.values)
        norm_train_features=normalizer_fit.transform(train_features.values)
        norm_test_features=normalizer_fit.transform(test_features.values)
    
    
    print(normalizer_fit.data_max_)
    print(normalizer_fit.data_min_)
    
    
    joblib.dump(normalizer_fit,filename_4_scalingfunction)
    print('saved joblib for scaling function as %s ', filename_4_scalingfunction)
    return norm_train_features,norm_test_features,normalizer_fit 



#Scale using training dataset on features
## data = np.random.rand(5, 4, 3)  # (samples, timesteps, features)

# Step 1: Reshape to 2D
## samples, timesteps, features = data.shape
## data_reshaped = data.reshape(-1, features)

# Step 2: Apply MinMaxScaler
## scaler = MinMaxScaler(feature_range=(-1, 1))
## data_scaled = scaler.fit_transform(data_reshaped)

# Step 3: Reshape back to 3D
## data_scaled_3d = data_scaled.reshape(samples, timesteps, features)

def scale_3D_ANN(data,data_test_features,filename_4_scalingfunction):
    # Step 1: Reshape Train dataset to 2D
    samples, timesteps, features = data.shape
    train_features = data.reshape(-1, features)
    
    # Step 2: Reshape Test dataset to 2D
    samples_test, timesteps_test, features_test = data_test_features.shape
    data_test_features = data_test_features.reshape(-1, features_test)
    
    normalizer = MinMaxScaler(feature_range=(-1,1))
    
    print(type(train_features))
    if isinstance(train_features,pd.core.series.Series):
        normalizer_fit = normalizer.fit(train_features.values.reshape(-1,1)) #Normalize based on training
        
        norm_train_features=normalizer_fit.transform(train_features.values.reshape(-1,1))
        norm_test_features=normalizer_fit.transform(data_test_features.values.reshape(-1,1))
    else:
        normalizer_fit = normalizer.fit(train_features) #Normalize based on training
        
        norm_train_features=normalizer_fit.transform(train_features)
        norm_test_features=normalizer_fit.transform(data_test_features)
    
    
    print(normalizer_fit.data_max_)
    print(normalizer_fit.data_min_)
    
    
    joblib.dump(normalizer_fit,filename_4_scalingfunction)
    print('saved joblib for scaling function as %s ', filename_4_scalingfunction)
    
    # Step 3: Reshape back to 3D
    norm_train_features = norm_train_features.reshape(samples, timesteps, features)
    norm_test_features = norm_test_features.reshape(samples_test, timesteps_test, features_test)
    
    return norm_train_features,norm_test_features,normalizer_fit 

def scale_3D_inference(data_test_features,normalizer_test):
    
    # Step 2: Reshape Test dataset to 2D
    samples_test, timesteps_test, features_test = data_test_features.shape
    data_test_features = data_test_features.reshape(-1, features_test) 
             
    norm_test_features=normalizer_test.transform(data_test_features)
        
    print(normalizer_test.data_max_)
    print(normalizer_test.data_min_)
        
    # normalizer_test=joblib.load(filename_4_scalingfunction)
    # print('loaded scaling function as %s ', filename_4_scalingfunction)
    
    # Step 3: Reshape back to 3D
    norm_test_features = norm_test_features.reshape(samples_test, timesteps_test, features_test)
    
    return norm_test_features


def create_training_data_lstm_4drilling1(training_set, lookback,errortraining=None): #Include Errortraining here.
  
    #Input shape of training set and error training is 2D ns by Q    
    print('training_shape',training_set.shape)
    m=training_set.shape[0]     
    Q=training_set.shape[1]
    Q1=errortraining.shape[1]
    training_set=training_set.to_numpy()
    print('sample',m,'Q_train',Q,'Q1_test',Q1)
    if errortraining is not None :
        print('error_training_shape',errortraining.shape)
        ytrain = [errortraining.iloc[i,:Q1] for i in range(lookback-1,m)] # errortraining.iloc[i,:Q1] , Remember zero-indexing so error corresponding to index i , i starts from lookback-1 and ends at m-1. select column error by selecting 0 (:1). Appends the list using list comprehension - Extract Q modes each time for m-lookback number of times
        #ytrain = [errortraining.iloc[i,:Q1] for i in range(m)]  #works for lookback equal to 1 and check for lookback>1
        
        ytrain = np.array(ytrain) #"2D array of size sample (m-lookback) x Grid size
        
        print('YTRAIN_SHAPE',ytrain.shape)        
        
    else:
        ytrain=None
               
    xtrain = np.zeros((m-lookback+1,lookback,Q))
    #xtrain_we = np.zeros((m-lookback+1,lookback,Q))
    
    #print(ytrain)
    for i in range(m-lookback+1): #(m-lookback): #number of samples is "m-lookback + 1", and for each sample/time index i (row of xtrain)
        #print(i)
        a = training_set[i,:Q] #if Q+1 then obtain variable values for features Q+1 for time index i (Q is grids, Q+1 for parameter value - time step here.)

        for j in range(1,lookback): # Obtain coefficients for time index i+j for look-back times foe each sample.
            #print(i,j)
            a = np.vstack((a,training_set[i+j,:Q])) # Add to row. concatenate in loop all feature Q values for different time steps,see the beautiful use of vstack to create for a given sample i, a 2D array of coefficients with lookback rows x Q mode columns 
               
        xtrain[i,:,:] = a  # 
        #xtrain_we[i,:,:]=a
        #if i>0:
            #A3=normalizer_fit.inverse_transform(xtrain_we[i,lookback-2,:].reshape(-1, 1))+ytrain[i-1,:] #y - error is not scaled, but feature values in xtrain_we is scaled. so we unscale it to add. 
            #print(A3,'A3shape')             
            #A2=normalizer_fit.transform(A3)
            #print(A2,'A2shape')
            #xtrain_we[i,lookback-2,:]=A2
    print('YTRAIN_SHAPE',ytrain.shape)         
    print('XTRAIN_SHAPE',xtrain.shape)   
    #print('XTRAINWE_SHAPE',xtrain_we.shape)       
    return xtrain,ytrain #xtrain_we  #,xtrain_3D,ytrain_1D #3D shape ns,lookback,1        

def testing_lstm_4drilling_nonrolling(training_set,lookback):      
    #Input shape of training set and error training is 2D ns by Q
    #print('training_shape',training_set.shape)
    m=training_set.shape[0]     
    Q=training_set.shape[1] 
    
    #print('Sample',m ,' Input Feature shape ',Q)     
               
    # Below: Obtain both predictions from model and create subsequent training data in Rolling LSTM format and Non-Rolling form.
          
    # Non-Rolling involves not correcting the input. Using uncorrected input for the next sample.
    
    xtrain = np.zeros((m-lookback+1,lookback,Q))
    # xtrain_we = np.zeros((m-lookback+1,lookback,Q))
    # ycorrect = np.zeros((m-lookback+1))
       
    for i in range(m-lookback+1): #(m-lookback): #number of samples is "m-lookback + 1", and for each sample/time index i (row of xtrain). On range the loop starts from 0 and ends at m-lookback.
        a = training_set[i,:Q] #if Q+1 then obtain variable values for features Q+1 for time index i (Q is grids, Q+1 for parameter value - time step here.)

        for j in range(1,lookback): # Obtain coefficients for time index i+j for look-back times foe each sample.
            #print(i,j)
            a = np.vstack((a,training_set[i+j,:Q])) # Add to row. concatenate in loop all feature Q values for different time steps,see the beautiful use of vstack to create for a given sample i, a 2D array of coefficients with lookback rows x Q mode columns 

        
        xtrain[i,:,:] = a
       
        
           
    #print('XTRAIN_j_SHAPE',xtrain.shape)      
                
    return xtrain      

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

