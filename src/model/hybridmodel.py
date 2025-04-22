"""
File: hybridmodel.py
Description: Contains the hybrid model class, including
            methods for training and prediction.


Author: Anndress07
Last update: 1/12/2024

Usage:
            Accessed by main.py
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict


class HybridModel:
    """
    Contains the structure for the entire hybrid model, the decision tree and each linear regression for the leaf nodes,
    as well as the methods necessary for training ( fit() ) and prediction ( predict() )
    The fit() method is meant to be used first with the training data as parameters (both X_train and y_train). It will
    generate all objects needed for the training.
    After calling fit(), predict() can be used with the X_test (or the data to predict). It will yield
    linear_predictions, a list with the output for each sample in X_test.

    """

    def __init__(self, **kwargs):
        self.decision_tree = DecisionTreeRegressor(**kwargs)
        self.is_fitted = False

        # self.leaf_params_dict: Holds the samples' parameters grouped by the leaf node they landed in
        self.leaf_params_dict = defaultdict(list)
        # self.leaf_result_dict: Holds the samples' target parameter, grouped by the leaf node they
        self.leaf_result_dict = defaultdict(list)

        self.linear_regression_results = dict()
        # self.y_pred = list()


    def fit(self, X_train, y_train):

        self.decision_tree.fit(X_train, y_train)
        self.is_fitted = True
        self.__classification(X_train, y_train)
        self.__regresor(X_train, y_train)


    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("The model is not fitted. Run fit() before predict()")

        leaf_node_test_list = self.decision_tree.apply(X_test)
        y_pred = []
        # for leaf_node in range(len(X_test)):
        #     leaf_id_value = leaf_node_test_list[leaf_node]
        #     for model in self.linear_regression_results:
        #         if model["Model: "] == leaf_id_value:
        #             current_linear_regressor_object = model["Object: "]
        #             current_x = X_test.iloc[leaf_node].to_frame().T.values
        #             y_pred_for_current_linear_regressor = current_linear_regressor_object.predict(current_X)
        #             self.y_pred.append(y_pred_for_current_linear_regressor[0])
        leaf_model_map = {
            model["model_id"]: model["object"]
            for model in self.linear_regression_results
        }

        for i, leaf_id in enumerate(leaf_node_test_list):
            linear_model = self.linear_regression_results.get(leaf_id)

            if linear_model is None:
                raise ValueError(f"No linear model found for leaf ID {leaf_id}")

            x_row = X_test.iloc[[i]]
            pred = linear_model.predict(x_row)[0]
            y_pred.append(pred)

        return y_pred

    def __classification(self, X_train, y_train):
        for leaf_node, leaf_id_value in enumerate(self.leaf_sample_list):
            self.leaf_params_dict[leaf_id_value].append(X_train.iloc[leaf_node].tolist())
            self.leaf_result_dict[leaf_id_value].append(y_train.iloc[leaf_node])

    def __regresor(self):

        for key, val in self.leaf_params_dict.items():
            local_linear_regressor = Ridge(alpha=1.0)  # Hardcoded Ridge as the linear regressor

            local_linear_regressor_X_train = self.leaf_params_dict[key]
            local_linear_regressor_y_train = self.leaf_result_dict[key]
            self.linear_regressor.fit(local_linear_regressor_X_train, local_linear_regressor_y_train)
            # local_linear_regressor_result = {"model_id": key, "object": local_linear_regressor}
            self.linear_regression_results[key] = local_linear_regressor
