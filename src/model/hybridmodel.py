"""
File: hybridmodel.py
Description: Contains the hybrid model class, including
            methods for training and prediction.

Author: marvincastroo
Last update: 26/4/2025


"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict


class HybridModel:

    def __init__(self, **kwargs):
        """

        :param kwargs: Hyperparameters for Scikit's DecisionTreeRegressor

        :var self.leaf_params_dict: Holds the samples' parameters, grouped by leaf node they landed in.
            { leaf_node_id : [ [params_for_sample1] ,..., [params_for_sampleN] ] }

        :var self.leaf_result_dict: Holds the samples' target feature, grouped by leaf node they landed in.
            { leaf_node_id : [y_for_sample1, ... , y_for_sampleN] }

        :var self.linear_regression_results: Holds the fitted linear regression for each leaf node in the tree.
            { leaf_node_id : ridge_object }

        """
        self.decision_tree = DecisionTreeRegressor(**kwargs)

        self.is_fitted = False
        self.leaf_params_dict = defaultdict(list)
        self.leaf_result_dict = defaultdict(list)
        self.linear_regression_results = dict()


    def fit(self, X_train, y_train):
        """
        Fits the hybrid model.
        :param X_train: (pd.Dataframe) training set parameters
        :param y_train: (pd.Dataframe) training set target value

        """
        self.decision_tree.fit(X_train, y_train)
        self.is_fitted = True
        self.__classification(X_train, y_train)
        self.__regressor()


    def predict(self, X_test):
        """
        Predicts.
        :param X_test: (pd.Dataframe) testing set parameters
        :return y_pred: (list) target values for the set in testing
        """
        if not self.is_fitted:
            raise ValueError("The model is not fitted. Run fit() before predict()")

        leaf_node_test_list = self.decision_tree.apply(X_test)
        y_pred = []

        for i, leaf_id in enumerate(leaf_node_test_list):
            _linear_model = self.linear_regression_results.get(leaf_id)

            if _linear_model is None:
                raise ValueError(f"No linear model found for leaf ID {leaf_id}")

            x_row = X_test.iloc[[i]].values
            pred = _linear_model.predict(x_row)[0]
            y_pred.append(pred)

        return y_pred

    def __classification(self, X_train, y_train):
        """
        Classifies each training sample in the decision tree's leaf nodes.
        Builds leaf_params_dict and leaf_result_dict dictionaries. Called during fit()
            self.leaf_params_dict Holds the samples' parameters, grouped by leaf node they landed in.
                { leaf_node_id : [ [params_for_sample1] ,..., [params_for_sampleN] ] }

            self.leaf_result_dict Holds the samples' target feature, grouped by leaf node they landed in.
                { leaf_node_id : [y_for_sample1, ... , y_for_sampleN] }

        :param X_train: (pd.Dataframe) training set parameters
        :param y_train: (pd.Dataframe) training set target value
        """
        leaf_sample_list = self.decision_tree.apply(X_train)
        for leaf_node, leaf_id_value in enumerate(leaf_sample_list):
            # self.leaf_params_dict[leaf_id_value].append(X_train.iloc[leaf_node].tolist())
            self.leaf_params_dict[leaf_id_value].append(X_train.iloc[leaf_node].values)
            self.leaf_result_dict[leaf_id_value].append(y_train.iloc[leaf_node])



    def __regressor(self):
        """
        Fits the linear regression in each leaf node of the tree. Builds the linear_regression_results dictionary.
            self.linear_regression_results Holds the fitted linear regression for each leaf node in the tree.
                { leaf_node_id : ridge_object }
        """

        for key, val in self.leaf_params_dict.items():
            local_linear_regressor = Ridge(alpha=1.0)  # Hardcoded Ridge as the linear regressor

            local_linear_regressor_X_train = np.array(self.leaf_params_dict[key])
            local_linear_regressor_y_train = np.array(self.leaf_result_dict[key])
            local_linear_regressor.fit(local_linear_regressor_X_train, local_linear_regressor_y_train)
            self.linear_regression_results[key] = local_linear_regressor

    def get_params(self, deep=True):
        """
        Returns the parameters for the decision tree.
        :param deep: (bool) If True, will return the parameters for this estimator and contained subobjects that are
            estimators.
        :return: (string) Parameter names mapped to their values.
        """
        return self.decision_tree.get_params(deep=deep)



