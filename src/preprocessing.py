import numpy as np
import pandas as pd

class Preprocessing:
    def __init__(self,
                 train = None,
                 test = None,
                 corner = None,
                 context_features = False,
                 std_dvt_context = False,
                 distance_parameter = True,
                 datasets_dict = None,
                 remove_nan = True
                 ):

        self.train_data = train
        self.test_data = test
        self.corner = corner
        self.context_features = context_features
        self.std_dvt_context = std_dvt_context
        self.distance_parameter = distance_parameter
        self.remove_nan = remove_nan
        self.initial_parameter_list = [' Fanout', ' Cap', ' Slew', ' Delay', 'X_drive', 'Y_drive', 'X_sink', 'Y_sink',
                                       'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context', 'σ(Y)_context',
                                       'Drive_cell_size', 'Sink_cell_size', 'Label Delay']
        self.datasets_dict = datasets_dict
        self.processed_datasets = {}

        self.train_data, self.test_data = self.preprocessing()


    def calculate_distance_parameter(self, *dfs):
        result = []

        for df in dfs:
            df['Distance'] = np.sqrt(
                (df["X_drive"] - df["X_sink"]) ** 2 + (df["Y_drive"] - df["Y_sink"]) ** 2
            )
            df = df.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])

            cols = df.columns.tolist()
            cols.remove('Distance')
            delay_idx = cols.index(' Delay')
            new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
            df = df[new_cols_order]

            result.append(df)

        return tuple(result)

    def preprocessing(self):
        if not self.context_features:
            self.initial_parameter_list.remove("X_context")
            self.initial_parameter_list.remove("Y_context")
        if not self.std_dvt_context:
            self.initial_parameter_list.remove("σ(X)_context")
            self.initial_parameter_list.remove("σ(Y)_context")

        if self.train_data is not None and self.test_data is not None:
            train_df = pd.read_csv(self.train_data)[self.initial_parameter_list]
            test_df = pd.read_csv(self.test_data)[self.initial_parameter_list]

            if self.distance_parameter:
                train_df, test_df = self.calculate_distance_parameter(train_df, test_df)
                if self.remove_nan:
                    train_df = train_df.dropna()
                    test_df = test_df.dropna()

            return train_df, test_df

        elif self.files_dict is not None:
            for output_name, input_file in self.datasets_dict.items():
                df = pd.read_csv(input_file)[self.initial_parameter_list]
                df = df.dropna() if self.remove_nan else df
                # df = df[self.initial_parameter_list]

                if self.distance_parameter:
                    df = self.calculate_distance_parameter(df)

                self.processed_datasets[output_name] = input_file

            return self.processed_datasets
        else:
            raise ValueError("train or test datasets, or dictionary of datasets not found. ")

    def to_csv(self, train_name = "processed_train", test_name = "processed_test"):
        if self.train_data is not None and self.test_data is not None:
            self.train_data.to_csv(f"../../datasets/processed/{train_name}.csv", index=False)
            self.test_data.to_csv(f"../../datasets/processed/{test_name}.csv", index=False)

        elif self.processed_datasets is not None:
            for output_name, input_file in self.processed_datasets.items():
                input_file.to_csv(f"../../datasets/processed/{output_name}.csv", index=False)
        else:
            raise ValueError("train or test datasets missing, or processed_datasets (dict) is None ")

    def get_data(self):
        if self.train_data is not None and self.test_data is not None:
            return self.train_data, self.test_data
        elif self.processed_datasets is not None:
            return self.processed_datasets
        else:
            raise ValueError("train or test datasets missing, or processed_datasets (dict) is None ")

