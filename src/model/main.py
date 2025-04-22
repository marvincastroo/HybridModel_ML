from src.model.hybridmodel import HybridModel
from src.preprocessing import Preprocessing
from matplotlib import pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split


pd.read_csv("../../datasets/unmodified/train.csv")




# hb_model = HybridModel(max_depth=9, max_features=15, random_state=10)
# hb_model.fit()


pre = Preprocessing(train="../../datasets/unmodified/train.csv",
                    test="../../datasets/unmodified/test_labels.csv",
                    )
pre.to_csv(train_name="first_iteration_train",
           test_name="first_iteration_test")
train_data, test_data = pre.get_data()




print(f"{train_data.columns=}")

# pre.to_csv()