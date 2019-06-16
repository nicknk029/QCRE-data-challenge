import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
 from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

######## ASSIGN THE ORGINAL EXCEL FILE TO 'data' DATAFRAME ###########

data = pd.read_excel('shifted_1.xlsx')   #### shifted_1.xlsx, the file which we got from data preparation ####
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

######### CREATE THE SUB MODELS #########

estimators = []
model_1 = LogisticRegression( max_iter = 10000, penalty = 'l1')
estimators.append( ( 'logistic', model_1 ) )

model_2 = DecisionTreeClassifier()
estimators.append( ( 'cart', model_2 ) )

model_3 = SVC()
estimators.append( ( 'svm', model_3 ) )

model_4 = KNeighborsClassifier( n_neighbors = 10 )
estimators.append( ( 'knn', model_4 ) )


######### CREATE THE ENSEMBLE MODEL ########

ensemble = VotingClassifier( estimators )
ensemble.fit( X_train_1, y_train_1 )
y_pred = ensemble.predict(X_test)

########### CLASSIFICATION REPORT, CONFUSION MATRIX & ACCURACY FOR EACH FOLD ##########

print( 'CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test, y_pred ), '\n' )
print( confusion_matrix( y_test, y_pred ), '\n' )
print( "ACCURACY :", accuracy_score( y_test, y_pred ), '\n' )
