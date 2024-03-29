######### IMPORT LIBRARIES ########

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

######## ASSIGN THE ORGINAL EXCEL FILE TO clean_train DATAFRAME ###########

clean_train = pd.read_excel( 'shifted_1.xlsx' ) #### shifted_1.xlsx, the file which we got from data preparation ####
cols=[ 'x%d' % i for i in range( 1, 123, 1 ) ]
cols.insert( 0,'y' )
clean_train.columns = [ cols ]
print( clean_train.head() )

######### ASSIGN DATASET VALUES ##########

y = clean_train['y'].values
x = clean_train.drop(['y'], axis = 1).values

######### PERFORM STRATIFIED K-FOLD ############

tr_fold = []
te_fold = []
skf = StratifiedKFold( n_splits=10 )

for train_index, test_index in skf.split( x, y ):

    print( "TRAIN:", train_index,'\n','TRAIN SHAPE:',train_index.size,'\n',"TEST:", test_index,'\n','TEST SHAPE:',test_index.size,'\n' )
    x_train, x_test = x[ train_index ], x[ test_index ]
    y_train, y_test = y[ train_index ], y[ test_index ]
    tr_fold.append( train_index )
    te_fold.append( test_index )

############ EXTRACT AND ASSIGN TRAIN DATA ###########

tr_fold_1 = clean_train.iloc[ tr_fold[0] ]
y_train_1 = tr_fold_1['y']
x_train_1 = tr_fold_1.iloc[:,1:123]

tr_fold_2 = clean_train.iloc[ tr_fold[1] ]
y_train_2 = tr_fold_2['y']
x_train_2 = tr_fold_2.iloc[:,1:123]

tr_fold_3 = clean_train.iloc[ tr_fold[2] ]
y_train_3 = tr_fold_3['y']
x_train_3 = tr_fold_3.iloc[:,1:123]

tr_fold_4 = clean_train.iloc[ tr_fold[3] ]
y_train_4 = tr_fold_4['y']
x_train_4 = tr_fold_4.iloc[:,1:123]

tr_fold_5 = clean_train.iloc[ tr_fold[4] ]
y_train_5 = tr_fold_5['y']
x_train_5 = tr_fold_5.iloc[:,1:123]

tr_fold_6 = clean_train.iloc[ tr_fold[5] ]
y_train_6 = tr_fold_6['y']
x_train_6 = tr_fold_6.iloc[:,1:123]

tr_fold_7 = clean_train.iloc[ tr_fold[6] ]
y_train_7 = tr_fold_7['y']
x_train_7 = tr_fold_7.iloc[:,1:123]

tr_fold_8 = clean_train.iloc[ tr_fold[7] ]
y_train_8 = tr_fold_8['y']
x_train_8 = tr_fold_8.iloc[:,1:123]

tr_fold_9 = clean_train.iloc[ tr_fold[8] ]
y_train_9 = tr_fold_9['y']
x_train_9 = tr_fold_9.iloc[:,1:123]

tr_fold_10 = clean_train.iloc[ tr_fold[9] ]
y_train_10 = tr_fold_2['y']
x_train_10 = tr_fold_2.iloc[:,1:123]

########### PERFORM SMOTE-OVERSAMPLING TECHNIQUE TO TRAIN DATA ###########

sm_1 = SMOTE()
x1_train_os, y1_train_os = sm_1.fit_sample( x_train_1, y_train_1.values.ravel() )

sm_2 = SMOTE()
x2_train_os, y2_train_os = sm_2.fit_sample( x_train_2, y_train_2.values.ravel() )

sm_3 = SMOTE()
x3_train_os, y3_train_os = sm_3.fit_sample( x_train_3, y_train_3.values.ravel() )

sm_4 = SMOTE()
x4_train_os, y4_train_os = sm_4.fit_sample( x_train_4, y_train_4.values.ravel() )

sm_5 = SMOTE()
x5_train_os, y5_train_os = sm_5.fit_sample( x_train_5, y_train_5.values.ravel() )

sm_6 = SMOTE()
x6_train_os, y6_train_os = sm_6.fit_sample( x_train_6, y_train_6.values.ravel() )

sm_7 = SMOTE() 
x7_train_os, y7_train_os = sm_7.fit_sample( x_train_7, y_train_7.values.ravel() )

sm_8 = SMOTE()
x8_train_os, y8_train_os = sm_8.fit_sample( x_train_8, y_train_8.values.ravel() )

sm_9 = SMOTE()
x9_train_os, y9_train_os = sm_9.fit_sample( x_train_9, y_train_9.values.ravel() )

sm_10 = SMOTE()
x10_train_os, y10_train_os = sm_10.fit_sample( x_train_10, y_train_10.values.ravel() )

########## EXTRACT AND ASSIGN TEST DATA ############

te_fold_1 = clean_train.iloc[ te_fold[0] ]
y_test_1 = te_fold_1['y']
x_test_1 = te_fold_1.iloc[:,1:123]

te_fold_2 = clean_train.iloc[ te_fold[1] ]
y_test_2 = te_fold_2['y']
x_test_2 = te_fold_2.iloc[:,1:123]

te_fold_3 = clean_train.iloc[ te_fold[2] ]
y_test_3 = te_fold_3['y']
x_test_3 = te_fold_3.iloc[:,1:123]

te_fold_4 = clean_train.iloc[ te_fold[3] ]
y_test_4 = te_fold_4['y']
x_test_4 = te_fold_4.iloc[:,1:123]

te_fold_5 = clean_train.iloc[ te_fold[4] ]
y_test_5 = te_fold_5['y']
x_test_5 = te_fold_5.iloc[:,1:123]

te_fold_6 = clean_train.iloc[ te_fold[5] ]
y_test_6 = te_fold_6['y']
x_test_6 = te_fold_6.iloc[:,1:123]

te_fold_7 = clean_train.iloc[ te_fold[6] ]
y_test_7 = te_fold_7['y']
x_test_7 = te_fold_7.iloc[:,1:123] 

te_fold_8 = clean_train.iloc[ te_fold[7] ]
y_test_8 = te_fold_8['y']
x_test_8 = te_fold_8.iloc[:,1:123]

te_fold_9 = clean_train.iloc[ te_fold[8] ]
y_test_9 = te_fold_9['y']
x_test_9 = te_fold_9.iloc[:,1:123]

te_fold_10 = clean_train.iloc[ te_fold[9] ]
y_test_10 = te_fold_10['y']
x_test_10 = te_fold_10.iloc[:,1:123]

########### PERFORM LOGISTIC REGRESSION #############

logreg1 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg1.fit( x1_train_os, y1_train_os )

logreg2 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg2.fit( x2_train_os, y2_train_os )

logreg3 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg3.fit( x3_train_os, y3_train_os )

logreg4 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg4.fit( x4_train_os, y4_train_os )

logreg5 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg5.fit( x5_train_os, y5_train_os )

logreg6 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg6.fit( x6_train_os, y6_train_os )

logreg7 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg7.fit( x7_train_os, y7_train_os )

logreg8 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg8.fit( x8_train_os, y8_train_os )

logreg9 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg9.fit( x9_train_os, y9_train_os )

logreg10 = LogisticRegression( class_weight = 'balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01 )
logreg10.fit( x10_train_os, y10_train_os )

########### CLASSIFICATION REPORT & CONFUSION MATRIX FOR EACH FOLD ##########

y_pred_1 = logreg1.predict( x_test_1 )
print( 'FOLD 1 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_1, y_pred_1 ), '\n' )
print( confusion_matrix( y_test_1, y_pred_1 ), '\n' )

y_pred_2 = logreg2.predict( x_test_2 )
print( 'FOLD 2 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_2, y_pred_2 ), '\n' )
print( confusion_matrix( y_test_2, y_pred_2 ), '\n' )

y_pred_3 = logreg3.predict( x_test_3 )
print( 'FOLD 3 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_3, y_pred_3 ), '\n' )
print( confusion_matrix( y_test_3, y_pred_3 ), '\n' )

y_pred_4 = logreg4.predict( x_test_4 )
print( 'FOLD 4 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_4, y_pred_4 ), '\n' )
print( confusion_matrix( y_test_4, y_pred_4 ), '\n' )

y_pred_5 = logreg5.predict( x_test_5 )
print( 'FOLD 5 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_5,y_pred_5 ), '\n' )
print( confusion_matrix( y_test_5, y_pred_5 ), '\n' )

y_pred_6 = logreg6.predict( x_test_6 )
print( 'FOLD 6 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_6, y_pred_6 ), '\n' )
print( confusion_matrix( y_test_6, y_pred_6 ), '\n' )

y_pred_7 = logreg7.predict( x_test_7 )
print( 'FOLD 7 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_7, y_pred_7 ), '\n' )
print( confusion_matrix( y_test_7, y_pred_7 ), '\n' )

y_pred_8 = logreg8.predict( x_test_8 )
print( 'FOLD 8 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_8, y_pred_8 ), '\n' )
print( confusion_matrix( y_test_8, y_pred_8 ), '\n' )

y_pred_9 = logreg9.predict( x_test_9 )
print( 'FOLD 9 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_9, y_pred_9 ), '\n' )
print( confusion_matrix( y_test_9, y_pred_9 ), '\n' )

y_pred_10 = logreg10.predict( x_test_10 )
print( 'FOLD 10 CLASSIFICATION REPORT & CONFUSION MATRIX ', '\n' )
print( classification_report( y_test_10, y_pred_10 ), '\n' )
print( confusion_matrix( y_test_10, y_pred_10 ), '\n')

######### ACCURACY FOR EACH FOLD ###########

print( "ACCURACY FOR FOLD 1:", accuracy_score( y_test_1, y_pred_1 ), '\n' )
print( "ACCURACY FOR FOLD 2:", accuracy_score( y_test_2, y_pred_2 ), '\n' )
print( "ACCURACY FOR FOLD 3:", accuracy_score( y_test_3, y_pred_3 ), '\n' )
print( "ACCURACY FOR FOLD 4:", accuracy_score( y_test_4, y_pred_4 ), '\n' )
print( "ACCURACY FOR FOLD 5:", accuracy_score( y_test_5, y_pred_5 ), '\n' )
print( "ACCURACY FOR FOLD 6:", accuracy_score( y_test_6, y_pred_6 ), '\n' )
print( "ACCURACY FOR FOLD 7:", accuracy_score( y_test_7, y_pred_7 ), '\n' )
print( "ACCURACY FOR FOLD 8:", accuracy_score( y_test_8, y_pred_8 ), '\n' )
print( "ACCURACY FOR FOLD 9:", accuracy_score( y_test_9, y_pred_9 ), '\n' )
print( "ACCURACY FOR FOLD 10:", accuracy_score( y_test_10, y_pred_10 ), '\n' )

########### PERFORM RANDOM FOREST #############

RF_1 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_1.fit( x1_train_os, y1_train_os )

RF_2 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_2.fit( x2_train_os, y2_train_os )

RF_3 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_3.fit( x3_train_os, y3_train_os )

RF_4 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_4.fit( x4_train_os, y4_train_os )

RF_5 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_5.fit( x5_train_os, y5_train_os )

RF_6 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_6.fit( x6_train_os, y6_train_os )

RF_7 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_7.fit( x7_train_os, y7_train_os )

RF_8 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_8.fit( x8_train_os, y8_train_os )

RF_9 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_9.fit( x9_train_os, y9_train_os )

RF_10 = RandomForestClassifier( n_estimators = 100, class_weight = 'balanced' )
RF_10.fit( x10_train_os, y10_train_os)

########### CLASSIFICATION REPORT & CONFUSION MATRIX FOR EACH FOLD ##########

y_pred_RF1 = RF_1.predict( x_test_1 )
print( 'FOLD 1 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_1, y_pred_RF1 ), '\n' )
print( confusion_matrix( y_test_1, y_pred_RF1 ), '\n' )

y_pred_RF2 = RF_2.predict( x_test_2 )
print( 'FOLD 2 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_2, y_pred_RF2 ), '\n' )
print( confusion_matrix( y_test_2, y_pred_RF2 ), '\n' )

y_pred_RF3 = RF_3.predict( x_test_3 )
print( 'FOLD 3 CLASSIFICATION REPORT & CONFUSION MATRIX','\n' )
print( classification_report( y_test_3, y_pred_RF3 ), '\n' )
print( confusion_matrix( y_test_3, y_pred_RF3 ), '\n' )

y_pred_RF4 = RF_4.predict( x_test_4 )
print( 'FOLD 4 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_4, y_pred_RF4 ), '\n' )
print( confusion_matrix( y_test_4, y_pred_RF4 ), '\n' )

y_pred_RF5 = RF_5.predict( x_test_5 )
print( 'FOLD 5 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_5, y_pred_RF5 ), '\n' )
print( confusion_matrix( y_test_5, y_pred_RF5 ), '\n' )

y_pred_RF6 = RF_6.predict( x_test_6 )
print( 'FOLD 6 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_6, y_pred_RF6 ), '\n' )
print( confusion_matrix( y_test_6, y_pred_RF6 ), '\n' )

y_pred_RF7 = RF_7.predict( x_test_7 )
print( 'FOLD 7 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_7, y_pred_RF7 ), '\n' )
print( confusion_matrix( y_test_7, y_pred_RF7 ), '\n' )

y_pred_RF8 = RF_8.predict( x_test_8 )
print( 'FOLD 8 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_8, y_pred_RF8 ), '\n' )
print( confusion_matrix( y_test_8, y_pred_RF8 ), '\n' )

y_pred_RF9 = RF_9.predict( x_test_9 ) 
print( 'FOLD 9 CLASSIFICATION REPORT & CONFUSION MATRIX', '\n' )
print( classification_report( y_test_9, y_pred_RF9 ), '\n' )
print( confusion_matrix( y_test_9, y_pred_RF9 ), '\n' )

y_pred_RF10 = RF_10.predict( x_test_10 )
print( 'FOLD 10 CLASSIFICATION REPORT & CONFUSION MATRIX ', '\n' )
print( classification_report( y_test_10, y_pred_RF10 ), '\n' )
print( confusion_matrix( y_test_10, y_pred_RF10 ), '\n' )

######### ACCURACY FOR EACH FOLD ###########

print( "ACCURACY FOR FOLD 1:", accuracy_score( y_test_1, y_pred_RF1 ), '\n' )
print( "ACCURACY FOR FOLD 2:", accuracy_score( y_test_2, y_pred_RF2 ), '\n' )
print( "ACCURACY FOR FOLD 3:", accuracy_score( y_test_3, y_pred_RF3 ), '\n' )
print( "ACCURACY FOR FOLD 4:", accuracy_score( y_test_4, y_pred_RF4 ), '\n' )
print( "ACCURACY FOR FOLD 5:", accuracy_score( y_test_5, y_pred_RF5 ), '\n' )
print( "ACCURACY FOR FOLD 6:", accuracy_score( y_test_6, y_pred_RF6 ), '\n' )
print( "ACCURACY FOR FOLD 7:", accuracy_score( y_test_7, y_pred_RF7 ), '\n' )
print( "ACCURACY FOR FOLD 8:", accuracy_score( y_test_8, y_pred_RF8 ), '\n' )
print( "ACCURACY FOR FOLD 9:", accuracy_score( y_test_9, y_pred_RF9 ), '\n' )
print( "ACCURACY FOR FOLD 10:", accuracy_score( y_test_10, y_pred_RF10 ), '\n' )
