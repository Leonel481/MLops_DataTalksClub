from typing import List, Tuple

from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from mlops.homework_03.utils.data_preparation.encoders import vectorize_features
from mlops.homework_03.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

#Modelo regresion lr
@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    Series,
    Series,
    Series,
    BaseEstimator,
]:
    df = data
    target = kwargs.get('target')

    # X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val , dv = vectorize_features(
        df
    )

    y_train = df[target]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)

    return X_train, dv, lr
