"""Use the Least Absolute Shrinkage and Selection Operator (LASSO) on the UCI
bike sharing dataset.
"""
import copy
import csv
import pprint
import random

import numpy as np
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.preprocessing


class Struct:
    """A convenient struct-like class.

    Source: Python Cookbook, Beazley and Jones. 3rd Edition. 8.11. Simplifying
        the Initialization of Data Structures.

    Usage:
        ```python
        class Stock(Structure):
            _fields = ['name', 'shares', 'price']

        s1 = Stock('ACME', 50, 91.1)
        s2 = Stock('ACME', 50, price=91.1)
        s3 = Stock('ACME', shares=50, price=91.1)
        ```
    """
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the remaining keyword arguments
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))

        # Check for any remaining unknown arguments
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))


class BikeShareDataset(Struct):
    """The bike sharing dataset."""
    _fields = ['data', 'target']


def _get_abs_err(dataset, reg):
    """Compute the absolute error of predictions w.r.t. targets."""
    return np.abs(reg.predict(dataset.data) - dataset.target).mean()


def _get_data(raw_data, indices):
    """Gets data and target, separated."""
    data = copy.deepcopy([raw_data[i] for i in indices])
    target = np.array(copy.deepcopy([d['cnt'] for d in data]),
                      dtype=np.float64)
    for datum in data:
        del datum['cnt']

    return data, target


def _get_mse(dataset, reg):
    """Compute the MSE of predictions w.r.t. targets."""
    return ((reg.predict(dataset.data) - dataset.target)**2).mean()


def lasso_bike_sharing():
    """LASSO some bikes."""
    with open('./bike-sharing/day.csv', 'r') as f:
        reader = csv.DictReader(f)
        raw_data = [row for row in reader]

    for datum in raw_data:
        # NOTE(brendan): Don't use the record index or date as covariates.
        del datum['instant']
        del datum['dteday']

        # NOTE(brendan): Additionally, we will ignore the count of casual and
        # registered users, for simplicity of predicting a single value (i.e.,
        # the count of all bikes shared: cnt).
        del datum['casual']
        del datum['registered']

        for key in ['temp',
                    'atemp',
                    'hum',
                    'windspeed',
                    'cnt']:
            datum[key] = float(datum[key])

    # NOTE(brendan): Take a random 20% subset of the data as the test set.
    num_test = round(0.2*len(raw_data))
    test_indices = random.sample(range(len(raw_data)), num_test)
    test_data, test_target = _get_data(raw_data, test_indices)

    train_indices = [i for i in range(len(raw_data)) if i not in test_indices]
    train_data, train_target = _get_data(raw_data, train_indices)

    vec = sklearn.feature_extraction.DictVectorizer()
    train_data = vec.fit_transform(train_data).toarray()
    test_data = vec.transform(test_data).toarray()

    pprint.pprint(vec.get_feature_names())

    scaler = sklearn.preprocessing.StandardScaler().fit(train_data)
    # train_data_scaled = scaler.transform(train_data)
    # test_data_scaled = scaler.transform(test_data)
    train_data_scaled = train_data
    test_data_scaled = test_data

    bike_train_dataset = BikeShareDataset(data=train_data_scaled,
                                          target=train_target)
    bike_test_dataset = BikeShareDataset(data=test_data_scaled,
                                         target=test_target)

    reg = sklearn.linear_model.LassoCV(cv=10)

    reg.fit(bike_train_dataset.data, bike_train_dataset.target)

    pprint.pprint(f'Train mean count: {train_target.mean()}')
    pprint.pprint(f'Train std count: {train_target.std()}')
    pprint.pprint(f'Train MSE: {_get_mse(bike_train_dataset, reg)}')
    pprint.pprint(f'Test MSE: {_get_mse(bike_test_dataset, reg)}')
    pprint.pprint(
        f'Train mean absolute error: {_get_abs_err(bike_train_dataset, reg)}')
    pprint.pprint(
        f'Test mean absolute error: {_get_abs_err(bike_test_dataset, reg)}')
    pprint.pprint(f'Optimal tuning parameter: {reg.alpha_}')

    coefficients = vec.inverse_transform(reg.coef_.reshape(1, -1))
    pprint.pprint(f'Coefficients: {coefficients}')

    train_keys = vec.get_feature_names()
    pprint.pprint('Zero coefficients:')
    pprint.pprint([key for key in train_keys
                   if key not in coefficients[0].keys()])


if __name__ == '__main__':
    lasso_bike_sharing()
