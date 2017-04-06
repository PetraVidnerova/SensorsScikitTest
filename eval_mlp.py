import pandas as pd 
from sklearn.utils import check_random_state 
import numpy as np 

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

rs = check_random_state(42)

def evaluate(taskname, train_name, test_name, model): 
    # load train data 
    data = pd.read_csv(train_name,sep=';')
    X_train = data[data.columns[:-1]]
    y_train = data[data.columns[-1]]
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    
    # load test data 
    data = pd.read_csv(test_name,sep=';') 
    X_test = data[data.columns[:-1]]
    y_test = data[data.columns[-1]]
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()

    #print(" *** Training *** ")
    history = model.fit(X_train, y_train,
                        batch_size=100, epochs=500,
                        verbose=0, validation_data=(X_test, y_test))

    yy_train = model.predict(X_train)
    yy_test = model.predict(X_test)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
        
    
    diff = (y_train-yy_train)
    train_error = 100*sum(diff*diff)/len(yy_train)
    
    diff = (y_test-yy_test)
    test_error = 100*sum(diff*diff)/len(yy_test) 

    print(taskname, train_error, test_error)
    
    
            
def create_model():
    model = Sequential()

    model = Sequential()
    model.add(Dense(100, input_shape=(8,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=[])
    
    return model 



if __name__ == "__main__":

    model = create_model()
    for x in ["CO", "NOx", "NO2", "C6H6", "NMHC"]:
    #for x in [ "CO" ]:
        for i in range(1,6):
            evaluate(x + "-" + str(i),
                     "data/"+x+"-nrm-part"+str(i)+".train.csv",
                     "data/"+x+"-nrm-part"+str(i)+".test.csv",
                     model)
                 
                 
                 

