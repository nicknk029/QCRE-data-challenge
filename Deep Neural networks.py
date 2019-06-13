######### IMPORT LIBRARIES ########

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,classification_report

######## ASSIGN THE ORGINAL EXCEL FILE TO 'data' DATAFRAME ###########

data = pd.read_excel('shifted_1.xlsx')
cols=[ 'x%d' % i for i in range( 1, 123, 1 ) ]
cols.insert( 0,'y' )
data.columns = [ cols ]
print( data.head() )

######### ASSIGN X & y ########

X = data.iloc[:,1:123].values
y = data['y'].values

######### TRAIN, TEST SPLIT ########

X_train,X_test,y_train,y_test=train_test_split( X, y, test_size=0.10, random_state = 1, stratify = y )

######### SMOTE-OVERSAMPLING TECHNIQUE #######

sm = SMOTE()
X_train_1, y_train_1 = sm.fit_sample( X_train, y_train.ravel() )

######### STANDARDIZATION ########

sc = StandardScaler()
X_train_tran = sc.fit_transform( X_train_1 )
X_test_tran = sc.transform( X_test )

######### NEURAL NETWORKS ########

classifier = Sequential()
classifier.add( Dense( 122, input_dim =122, activation='relu', kernel_initializer = 'uniform' ) )
classifier.add( Dense( 122, activation='relu', kernel_initializer='uniform' ) )
classifier.add( Dense( 61, activation='relu', kernel_initializer='uniform' ) )
classifier.add( Dense( 2, activation='softmax', kernel_initializer='uniform' ) )
classifier.compile( optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'] )
classifier.fit( X_train_tran, y_train_1, epochs=10, batch_size = 100 )

######### CLASSIFICATION REPORT & CONFUSION MATRIX ########

predictions=classifier.predict( X_test_tran )
predictions_classes=classifier.predict_classes( X_test_tran )
print( 'Confusion matrix:', confusion_matrix( y_test, predictions_classes ) )
print( 'Classification report:', classification_report( y_test, predictions_classes ) )