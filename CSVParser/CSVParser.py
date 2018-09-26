import numpy as np
import pandas as pd

def concatenate_matrixes(base,toAdd):
    if base is None:
        return toAdd
    else:
        return np.c_[base, toAdd]

def csv_parser(data):
    result_X=None
    result_Y=None
    resultColumn='Нефть, т' # Y column
    parameterIndexes={} # dictionary[columnName, index_in_resultX]
    index=0
    for columnName in data.columns:
        #skip unwanted columns
        if columnName=='Нефть, м3' or columnName=='Жидкость, м3' or columnName=='Дебит жидкости' or columnName=='ТП(ИДН) Дебит жидкости' or columnName=='ТП(ИДН) Дебит жидкости скорр-ый' or columnName=='ТП(ГРП) Дебит жидкости' or columnName=='ТП(ГРП) Дебит жидкости скорр-ый' or columnName=='ГП - Общий прирост Qн':
            continue
        column=data[columnName].fillna(0)
        # fill Y and continue loop
        if columnName==resultColumn:
            result_Y=pd.to_numeric(column.str.replace(',','.')).fillna(0).values
            continue
        # fill X
        if  column.dtype != np.dtype(getattr(np, 'int64')) and column.dtype != np.dtype(getattr(np, 'float64')):
            # try to convert non numeric columns to numeric
            column = column.str.replace(',','.')
            try:
                floatVals=pd.to_numeric(column).fillna(0).values
                result_X = concatenate_matrixes(result_X, floatVals)
                # X column has been added -> add record to parameterIndexes
                if (columnName!= resultColumn):
                    parameterIndexes[columnName] = index
                    index+=1
            except:       
                # failed -> skip this column
                pass
        else:
            result_X = concatenate_matrixes(result_X, column.values)
            # X column has been added -> add record to parameterIndexes
            if (columnName!= resultColumn):
                parameterIndexes[columnName] = index
                index+=1
    return result_X, result_Y, parameterIndexes
