
####### IMPORT NECESSARY LIBRARIES #########

import pandas as pd
import numpy as np

######## ASSIGN THE ORIGINAL EXCEL FILE TO raw_train DATAFRAME ##########

raw_train = pd.read_excel( 'processminer-rareevent.xlsx' )
print( raw_train.head() )

############ SHIFT THE y COLUMN UP BY 1 ROW #############

x_cols = raw_train.drop( ['time', 'y'], axis = 1 )
X_col = x_cols.drop( x_cols.index[len(x_cols)-1] )
y_col = pd.DataFrame( raw_train['y'].shift(-1) )
y_cols = y_col.dropna()
shift_train = pd.concat( [y_cols, X_col], axis = 1 )

############## CLEAN TRAIN DATA - FORM 1 ##############
###### IN FORM 1 THE THE FIRST 61 COLUMNS HAVE i - (i-1) ROW VALUES AND THE SECOND 61 COLUMNS HAVE i -(i-2) ###########
########### CONSIDERED CYCLE BREAKS ############

a = []
b = []
c = []

for i in range( len( shift_train ) ):

    if i < 2:
        continue

    if shift_train['y'][i] == 0:
        
        if shift_train['y'][i-1] == 1 or shift_train['y'][i-2] == 1:
            continue

        else:
            a.append( y_col.values[i] )
            b.append( x_cols.values[i] - x_cols.values[i-1] )
            c.append( x_cols.values[i] - x_cols.values[i-2] )
            
    else:
        a.append( y_col.values[i] )
        b.append( x_cols.values[i] - x_cols.values[i-1] )
        c.append( x_cols.values[i] - x_cols.values[i-2] )

cl_1 = pd.DataFrame( a )
cl_2 = pd.DataFrame( b )
cl_3 = pd.DataFrame( c )
clean_data_nn = pd.concat( [cl_1, cl_2, cl_3], axis = 1 )

############# WRITING TO EXCEL ################

writer = pd.ExcelWriter( 'shifted_1.xlsx' )
clean_data_nn.to_excel( writer, 'Sheet1', index = False )
writer.save()

############## CLEAN TRAIN DATA - FORM 2 ##############
########## IN FORM 2 THE THE FIRST 61 COLUMNS HAVE i - (i-1) ROW VALUES AND THE SECOND 61 COLUMNS HAVE (i-1) -(i-2) #############
########### CONSIDERED CYCLE BREAKS ############

a = []
b = []
c = []

for i in range( len( shift_train ) ):

    if i < 2:
        continue

    if shift_train['y'][i] == 0:
        
        if shift_train['y'][i-1] == 1 or shift_train['y'][i-2] == 1:
            continue

        else:
            a.append( y_col.values[i] )
            b.append( x_cols.values[i] - x_cols.values[i-1] )
            c.append( x_cols.values[i-1] - x_cols.values[i-2] )

    else:
        a.append( y_col.values[i] )
        b.append( x_cols.values[i] - x_cols.values[i-1] )
        c.append( x_cols.values[i-1] - x_cols.values[i-2] )

cl_1 = pd.DataFrame( a )
cl_2 = pd.DataFrame( b )
cl_3 = pd.DataFrame( c )
clean_data_nn = pd.concat( [cl_1, cl_2, cl_3], axis = 1 )

############# WRITING TO EXCEL ################

writer = pd.ExcelWriter( 'shifted_2.xlsx' )
clean_data_nn.to_excel( writer, 'Sheet1', index = False )
writer.save()

############### CLEAN TRAIN DATA - FORM 3 ###################
############### IN FORM 3 THE THE FIRST 61 COLUMNS HAVE i - (i-1) ROW VALUES, SECOND 61 COLUMNS HAVE (i-1) -(i-2) AND THIRD 61 COLUMNS HAVE i - (i-2) ##############
########### CONSIDERED CYCLE BREAKS ################

a = []
b = []
c = []
d = []

for i in range( len( shift_train ) ):

    if i < 2:
        continue

    if shift_train['y'][i] == 0:

        if shift_train['y'][i-1] == 1 or shift_train['y'][i-2] == 1:
            continue

        else:
            a.append( y_col.values[i] )
            b.append( x_cols.values[i] - x_cols.values[i-1] )
            c.append( x_cols.values[i-1] - x_cols.values[i-2] )
            d.append( x_cols.values[i] - x_cols.values[i-2] )

    else:
        a.append( y_col.values[i] )
        b.append( x_cols.values[i] - x_cols.values[i-1] )
        c.append( x_cols.values[i-1] - x_cols.values[i-2] )
        d.append( x_cols.values[i] - x_cols.values[i-2] )

cl_1 = pd.DataFrame( a )
cl_2 = pd.DataFrame( b )
cl_3 = pd.DataFrame( c )
cl_4 = pd.DataFrame( d )
clean_data_nn = pd.concat( [cl_1, cl_2, cl_3, cl_4], axis = 1 )

############# WRITING TO EXCEL ################

writer = pd.ExcelWriter( 'shifted_3.xlsx' )
clean_data_nn.to_excel( writer, 'Sheet1', index = False )
writer.save()

###### TOTAL THERE WILL BE THREE NEW EXCEL FILES ########
