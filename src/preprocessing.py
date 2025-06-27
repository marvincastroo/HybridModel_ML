import numpy as np
import pandas as pd
import pickle
from typing import Literal, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Preprocessing:
    def __init__(self,
                 corner: Optional[Literal["fast", "slow"]] = None,
                 feature_scaling: Optional[Literal["standard", "minmax"]] = None,
                 context_features = False,
                 std_dvt_context = False,
                 distance_parameter = True,
                 remove_nan = True,
                 test_design_name = None,
                 verbose = 1,
                 ):


        self.corner = corner
        self.feature_scaling = feature_scaling
        self.test_name = test_design_name
        self.context_features = context_features
        self.std_dvt_context = std_dvt_context
        self.distance_parameter = distance_parameter
        self.remove_nan = remove_nan
        self.verbose = verbose
        self.filepaths_name = []
        self.processed_dfs = []




    def load_data(self, file_paths):
        self.filepaths_name = file_paths
        # for fp, index in enumerate(file_paths):
        #     # if i == 0:
        #     print(f"{fp=}, {index=}")
        self.processed_dfs = [self.preprocessing(fp, index) for index, fp in enumerate(file_paths)]

        self.__print_info()

    def corners(self, df):

        if self.corner.lower().replace(" ", "")== 'fast':
            fast_df = df.loc[df.groupby(df.columns[:].tolist())['Label Delay'].idxmin()] # todo: check [:]
            fast_df = fast_df.reset_index(drop=True)
            return fast_df

        elif self.corner.lower().replace(" ", "")== 'slow':
            slow_df = df.loc[df.groupby(df.columns[:].tolist())['Label Delay'].idxmax()]
            slow_df = slow_df.reset_index(drop=True)
            return slow_df

        # TODO: typical filtering not working properly.
        # elif self.corner.lower().replace(" ", "") == "typical":
        #     grouped = df.groupby(df.columns[:16].tolist())
        #     filtered_df = grouped.apply(get_quantile_row, quantile_value=0.5).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown corner value '{self.corner}'. Currently, only 'fast' and 'slow' corners are "
                             f"supported.")
            return

    def scale_dataframes(self, df, index):
        # first dataframe should be the training set, who will scale the following sets
        if index == 0:
            if self.feature_scaling == 'standard':
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df)
                df_scaled = pd.DataFrame(scaled, columns = df.columns)
                with open('../temp/standard_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)


            elif self.feature_scaling == 'minmax':
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df)
                df_scaled = pd.DataFrame(scaled, columns=df.columns)
                with open('../temp/minmax_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)

            else:
                raise ValueError(f"Unknown feature_scaling value '{self.feature_scaling}'. Currently, only 'standard' "
                                 f"and 'minmax' values are "
                                 f"supported.")

            return df_scaled

        # if the df is not the first one in the list (or, is a test set)
        else:
            if self.feature_scaling == 'standard':
                with open('../temp/standard_scaler.pkl', 'rb') as f:
                    std_scaler = pickle.load(f)
                test_scaled = std_scaler.transform(df)
                df_test_scaled = pd.DataFrame(test_scaled, columns=df.columns)
                return df_test_scaled

            elif self.feature_scaling == 'minmax':
                with open('../temp/minmax_scaler.pkl', 'rb') as f:
                    minmax_scaler = pickle.load(f)
                test_scaled = minmax_scaler.transform(df)
                df_test_scaled = pd.DataFrame(test_scaled, columns=df.columns)
                return df_test_scaled

        raise ValueError("Unexpected exit")
        return


    def __calculate_distance_parameter(self, df):


        df['Distance'] = np.sqrt(
            (df["X_drive"] - df["X_sink"]) ** 2 + (df["Y_drive"] - df["Y_sink"]) ** 2
        )
        df = df.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])

        cols = df.columns.tolist()
        cols.remove('Distance')
        delay_idx = cols.index(' Delay')
        new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
        df = df[new_cols_order]

        return df

    def preprocessing(self, data_filepath, index):
        # index is i-th element on the list of dataframes to process. This way we can differ the first df from others
        initial_parameter_list = [' Fanout', ' Cap', ' Slew', ' Delay', 'X_drive', 'Y_drive', 'X_sink', 'Y_sink',
                                  'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context', 'σ(Y)_context',
                                  'Drive_cell_size', 'Sink_cell_size', 'Label Delay', 'Design']



        if not self.context_features:
            initial_parameter_list.remove("X_context")
            initial_parameter_list.remove("Y_context")
        if not self.std_dvt_context:
            initial_parameter_list.remove("σ(X)_context")
            initial_parameter_list.remove("σ(Y)_context")

        df = pd.read_csv(data_filepath)

        if self.corner is not None:
            df = self.corner(df)




        # df = df.head(5) # TODO: REMOVE THIS
        cols_to_keep = [col for col in initial_parameter_list if col in df.columns]
        df = df[cols_to_keep]
        if self.test_name is not None:
            if "Design" in cols_to_keep:
                    df = df[df["Design"] == self.test_name]

            if df.empty:

                raise ValueError(f"The design {self.test_name} is not in the file {data_filepath}. \n"
                                 f"Remove the parameter 'test_design_name' or make sure the design exists inside the file.  ")
        if 'Design' in df.columns:
            df = df.drop(columns=["Design"])

        if self.feature_scaling is not None:
            df = self.scale_dataframes(df, index)


        if self.distance_parameter:
            df = self.__calculate_distance_parameter(df)
        if self.remove_nan:
            df = df.dropna()

        # else:
        #     raise ValueError("train or test datasets, or dictionary of datasets not found. ")
        return df



    def to_csv(self, file_names = None, directory="../datasets/processed/"):
        for i, df in enumerate(self.processed_dfs):
            name = file_names[i] if file_names and i < len(file_names) else f"file_{i}.csv"
            full_path = f"{directory.rstrip('/')}/{name}"
            df.to_csv(full_path, index=False)

    def get_data(self):
        return tuple(self.processed_dfs)
    def __print_messages(self, message, message_level):
        if message_level <= self.verbose:
            print(message)



    def __print_info(self):
        self.__print_messages("Processing class initiated", 1)
        self.__print_messages(f"\t-{self.corner=}", 2)
        self.__print_messages(f"\t-{self.context_features=}", 2)
        self.__print_messages(f"\t-{self.std_dvt_context=}", 2)
        self.__print_messages(f"\t-{self.remove_nan=}", 2)
        self.__print_messages(f"\t-{self.test_name=}", 2)
        self.__print_messages(f"\n", 2)

        for i, df in enumerate(self.processed_dfs):
            self.__print_messages(f"\t\tDataset #{i}:", 2)
            self.__print_messages(f"\t\t\tName: {self.filepaths_name[i]}:", 2)
            self.__print_messages(f"\t\t\tShape: {df.shape}:", 2)
            self.__print_messages(f"{df.head(5)}:", 2)

