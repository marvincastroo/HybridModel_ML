from src.model.hybridmodel import HybridModel
from src.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import root_mean_squared_error
import time

pre = Preprocessing(#test_design_name ="S15850",
                    verbose=2,
                    remove_nan=True,
                    feature_scaling='standard'
                    )

pre.load_data(["../datasets/processed/slow.csv",
               "../datasets/processed/designs_slow.csv"])

train_data, test_data1 = pre.get_data()

pre.to_csv(file_names=['test_scaling_train.csv', "test_scaling_test1.csv"])

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

X_test = test_data1.iloc[:, :-1]
y_test = test_data1.iloc[:, -1]
# print(f"{X_test=}")
# print(f"{y_test=}")

X_train, _, y_train, _ = train_test_split(X, y, random_state=10, test_size=0.2)
"""
tree params={'ccp_alphgit sa': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 
None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 
'random_state': None, 'splitter': 'best'}
"""
hb_model = HybridModel(max_depth=9,
                       max_features=15,
                       random_state=10)

hb_model.fit(X_train, y_train)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

start = time.time()
y_pred = hb_model.predict(X_test)
print(f"elapsed time: {time.time() - start}")

# hb_model2 = HybridModel2()
# hb_model2.fit(X_train=X_train, y_train=y_train, LR_type=1, custom_params=[13,13])
# start = time.time()
# y_pred2 = hb_model2.predict(X_test)
# print(f"elapsed time: {time.time() - start}")


# print(f"{y_pred=}")

openlane_predictions = X_test[" Delay"]

opl_rmse = root_mean_squared_error(openlane_predictions, y_test)
hybrid_rmse = root_mean_squared_error(y_pred, y_test)
print(f"{opl_rmse=}")
print(f"{hybrid_rmse=}")
