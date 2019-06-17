import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import warnings
import logging
import time
from joblib import dump
import pandas as pd
import argparse


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info("Time taken = {}".format(te-ts))
        return result
    return timed


class HitsRegression:

    def __init__(self):
        """
        Initialization method where the rmse is arbitrarily set to Infinity
        """
        self.root_mean_squared_error = np.inf

    def dispose(self):
        """
        Deletes this instance from memory
        :return: None
        """
        del(self)
        return None

    def get_scores(self,y_true, y_pred):
        """
        This function calculates some popular metrics for regression by
        comparing the true and predicted values
        :param y_true: the ground true reference
        :param y_pred: the predictions
        :return: None
        """
        try:
            assert(y_true.shape[0]==y_pred.shape[0])
            # making a combined dataframe
            df = pd.DataFrame({"row_num":y_true["row_num"].values, "true":y_true["distributed_hits"].values, "pred":y_pred[:,0]})
            # aggregating scores for each row number
            df = df.groupby("row_num").sum()
            # calculating scores
            self.root_mean_squared_error = np.sqrt(mse(df["true"],df["pred"]))
        except AssertionError as error:
            logging.error("Unequal number of observations")

    @timeit
    def get_data(self,path,max_rows=None):
        """
        This function reads data from disk
        :param path: A path on the disk
        :return: raw data of type pandas dataframe
        """
        try:
            raw_data = pd.read_csv(path,delimiter=';',nrows=max_rows)
            return raw_data
        except:
            logging.error("Invalid path")

    def get_plots(self,y_true,y_pred):
        """
        This function plot the predictions against true data points
        :param y_true: true values
        :param y_pred: predictions
        :return: None
        """
        try:
            assert(y_true.shape[0]==y_pred.shape[0])
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            xc = np.arange(y_true.shape[0])
            ax1.plot(xc, y_pred, label='pred')
            ax1.plot(xc, y_true, label='true')
            ax1.set_ylabel("Hits")
            plt.legend()
            plt.show()
            return None
        except AssertionError as error:
            logging.error("Unequal number of samples in output")

    def get_distributed_hits(self,data):
        """
        this function creates a column/series to divide the number of hits uniformly for each page the user has visited
        :param data: the input dataframe
        :return: series/column of distributed hits
        """
        data["distributed_hits"] = data.apply(lambda row: float(row.revised_hits) / len(str(row.path_id_set).split(";")), axis=1)

        return data

    def get_path_id(self, data):
        """
        this function creates a column/series to explode the path_id_set column into multiple rows
        :param data: the input dataframe
        :return: series/column of path_id
        """

        try:

            def perform_split(value, separator):
                """
                this is an auxiliary splitting function
                :param value: the input (non-)string
                :param separator: the splitter/delimiter
                :return: list of strings
                """
                string = str(value)
                if string == "":
                    return [0]
                else:
                    return string.split(separator)

            s1 = data["path_id_set"].apply(lambda x: perform_split(x, ";"))  # convert to string if not already
            s2 = s1.apply(pd.Series, 1).stack().reset_index(level=1, drop=True)
            s2.name = "path_id"
            data = data.join(s2)

            return data
        except:
            logging.error("check input")

    def get_total_locations(self,data):
        """
        this is a function to count the total locations visited in a particular session
        :param data: the input data frame
        :return: data frame with a new column having the count
        """

        def perform_split(string,separator):
            """
            this is an auxiliary splitting function
            :param value: the input (non-)string
            :param separator: the splitter/delimiter
            :return: list of strings
            """
            if string == "":
                return 0
            else:
                return len(string.split(separator))


        data["number_of_locations_visited"] = data.apply(lambda row: perform_split(str(row.path_id_set),";"), axis=1)
        return data

    def get_hits_and_sessions_fractions(self,data,observed_data,list_of_features):
        """
        this function calculates the fraction of times a feature was used for visiting
        and the fraction of hits it has contributed
        :param data: data to be added
        :param observed_data: data for which number of hits are given
        :param list_of_features: the features to be observed
        :return: enriched dataframe
        """
        for ftr in list_of_features:
            sessions_per_feature = data.groupby(ftr)["row_num"].count()
            hits_per_feature = observed_data.groupby(ftr)["revised_hits"].sum()
            df_fraction = pd.DataFrame({"total_sessions": sessions_per_feature, "total_hits": hits_per_feature})

            total_hits = df_fraction["total_hits"].sum()
            total_sessions = df_fraction["total_sessions"].sum()

            df_fraction[ftr+"_session_fraction"] = df_fraction.apply(lambda row: float(row.total_sessions)/total_sessions, axis=1)
            df_fraction[ftr+"_hits_fraction"] = df_fraction.apply(lambda row: float(row.total_hits)/total_hits, axis=1)

            df_fraction = df_fraction.drop(["total_sessions","total_hits"],axis=1)
            data = data.join(df_fraction, on=ftr)

        # check for null values and fill them with zero if present
        data = data.fillna(0)

        return data

    def get_revised_hits(self,data):
        """
        this function helps to clean the 'hits' column
        :param data: original data frame
        :return: new data frame with a 'revised hits' column
        """
        def clean(value):
            """
            this is an auxiliary function to check and substitute values
            :param value: the hits value
            :return: hits value casted to integer
            """
            if value == "\\N":
                return -1
            else:
                return int(value)

        data["revised_hits"] = data.apply(lambda row:clean(row.hits),axis=1)
        return data

    def clean_session_duration(self,data):
        """
        this function cleans the column - 'session duration'
        :param data: the original dataframe
        :return: data frame with the 'session duration' column cleaned
        """
        def clean(value):
            """
            this function checks for null values
            :param value: session duration
            :return: integer
            """
            if value.session_duration == "\\N":
                value.session_duration = 0
            else:
                pass

            return value

        data = data.apply(lambda row:clean(row),axis=1)
        return data

    def pre_process(self,data):
        """
        this function is responsible to prepare the dataset for training and testing
        :param data: the raw data
        :return: standardised training input and output data, test input and output data, validation input data
        and a scaling object of the training-output data
        """

        try:
            assert isinstance(data,pd.DataFrame),"data is not of type pandas.Dataframe"
            logging.info("Pre processing raw data")

            logging.info("preliminary cleaning")

            # clean 'session duration' column
            data = self.clean_session_duration(data)

            # the rows having negative hits will help us identify as the validation dataset
            # the validation data set is the one used to create the final results for submission
            # get revised hits
            data = self.get_revised_hits(data)


            logging.info("staring feature engineering")

            # distribute hits for each path id
            data = self.get_distributed_hits(data)


            # get total number of locations visited
            data = self.get_total_locations(data)


            list_of_categorical_features = ['locale','day_of_week','agent_id','traffic_type','hour_of_day']



            # get fraction of hits and sessions
            list_of_features_for_fraction = ["entry_page"]+list_of_categorical_features
            data = self.get_hits_and_sessions_fractions(data,data.loc[data["distributed_hits"] >= 0],list_of_features_for_fraction)

            # explode path id for each row into multiple rows
            data = self.get_path_id(data)
            data.reset_index(drop=True, inplace=True)

            # converting categorical columns to one-hot-encoded form into multiple columns
            categorical_dummy_df_list = []
            for ftr in list_of_categorical_features:
                categorical_dummy_df_list.append(pd.get_dummies(data[ftr]))



            # scaling non categorical columns - entry page, path id, session duration and number of locations visited
            list_of_non_categorical_features = ["path_id", "entry_page", "session_duration", "number_of_locations_visited"]
            array_of_non_categorical_features = data[list_of_non_categorical_features].values

            scaled_array = MinMaxScaler(feature_range=(0.05, 0.95)).fit_transform(
                StandardScaler().fit_transform(array_of_non_categorical_features))

            scaled_df = pd.DataFrame({"new_path_id": scaled_array[:, 0],
                                      "new_entry_page": scaled_array[:, 1],
                                      "new_session_duration": scaled_array[:, 2],
                                      "new_number_of_locations_visited":scaled_array[:,3]})

            # removing un-necessary columns
            list_of_features_to_remove = list_of_categorical_features+list_of_non_categorical_features+["revised_hits","hits","path_id_set"]
            data = data.drop(list_of_features_to_remove, axis=1)


            # attaching the new columns to the original data
            for dummy_df in categorical_dummy_df_list:
                data = pd.concat([data,dummy_df],axis=1)

            data = pd.concat([data, scaled_df],axis=1)

            logging.info("feature engineering complete")

            # ensure that the data is clean
            assert(np.any(np.isnan(data.values))==False)
            assert(np.any(np.isinf(data.values))==False)


            # separating the validation and train-test data sets
            data_train_test = data.loc[data["distributed_hits"] >= 0]
            data_validation = data.loc[data["distributed_hits"] < 0]


            # separating input and output columns
            input_data_train_test = data_train_test.copy().drop(["distributed_hits","row_num"],axis=1)
            output_data_train_test = data_train_test[["row_num","distributed_hits"]]
            input_data_validation = data_validation.drop("distributed_hits",axis=1)

            # shuffle and split train and test inputs and outputs
            x_train, x_test, y_train, y_test = train_test_split(input_data_train_test, output_data_train_test,shuffle=True)

            # initialising the scalers
            scaler_y = StandardScaler()

            # fitting scalers
            scaler_y.fit(y_train["distributed_hits"].values.reshape(-1,1))

            # attaching row_num with the validation input dataset
            x_val = input_data_validation.values

            # scaling output dataset
            y_train_std = scaler_y.transform(y_train["distributed_hits"].values.reshape(-1,1))

            return x_train, y_train_std, x_test, y_test, x_val, scaler_y
        except AssertionError as error:
            print("NaN or Inf in main dataset")
			
			
    @timeit
    def train(self,x_train,y_train):
        """
        initiating and fitting an ML model
        :param x_train: training input
        :param y_train: training output
        :return: trained model
        """
        try:
            assert (x_train.shape[0]==y_train.shape[0])
            logging.info("Training model")
            model = MLPRegressor(hidden_layer_sizes=[100, 100],activation="relu")
            #model = RandomForestRegressor()
            #model = DecisionTreeRegressor()
            model.fit(x_train, y_train)
            return model
        except AssertionError as error:
            logging.error("Unequal number of samples")

    def post_process(self,y_pred,scaler):
        """
        inverting the predictions to their original scale and rounding them to the nearest integer
        :param y_pred: raw predictions
        :param scaler: scaling object for the output data
        :return: transformed predictions
        """
        logging.info("Post processing predictions")
        y_processed = scaler.inverse_transform(y_pred.reshape(-1,1))
        return y_processed

    def predict(self,input):
        """
        A wrapper function gives predictions for a certain input.
        :param input: (test) input
        :return: predictions
        """
        logging.info("Predicting")
        output = self.model.predict(input)
        return output

    def get_final_submission_file(self,validation_input,scaler_y):
        """
        this function makes the submission file
        :param validation_input: the (pre-processed) rows in the dataset where there is no count on hits
        :param scaler_y: the scale of the output dataset with which the training-outputs were transformed
        :return:
        """
        # predicting on the validation data-set
        validation_output_raw = self.predict(validation_input[:,1:])

        # transforming predictions to its original scale
        validation_output = self.post_process(validation_output_raw,scaler_y)

        # making a dataframe for the predicted values
        validation_df = pd.DataFrame({"row_num":validation_input[:,0], "distributed_hits":validation_output[:,0]})

        # casting the row numbers to integer
        validation_df.row_num = validation_df.row_num.astype(int)

        # aggregating the hits for each row number to get total 'hits'
        final_df = validation_df.groupby("row_num").sum()

        # rounding the hits and casting them back to integer as in the original dataset
        final_df.distributed_hits = final_df.distributed_hits.round().astype("int")

        # renaming column from 'distributed_hits' to 'hits' as in the original dataset
        final_df = final_df.rename(columns={"distributed_hits":"hits"})

        # writing the results to a CSV file with ";" as the separator as in the original dataset
        final_df.to_csv("submission.csv",sep=';')
        return None

    def main(self,path,n_rows,make_submission,plot=False,threshold_rmse=100):
        """
        this is the main function that initiates the pipeline
        :param path: the path on the disk where the data is kept
        :param n_rows: the path on the disk where the data is kept
        :param make_submission: to write the submissions file or not
        :param plot: plot the predictions on the test dataset against the true values
        :param threshold_rmse: the minmum error needed to persist the model on the disk
        :return: None
        """
        try:
            assert(path!="")
            logging.info("Starting pipeline")

            # fetching the data
            data = self.get_data(path,max_rows=n_rows)

            # pre processing the data
            x_train, y_train, x_test, y_test, x_val, scaler_y = self.pre_process(data)

            # training
            self.model = self.train(x_train,y_train)

            # making predictions on the transformed dataset
            y_pred_raw = self.predict(x_test)

            # inverting the predictions to their original scale
            y_pred = self.post_process(y_pred_raw,scaler_y)

            # generating scores
            self.get_scores(y_test,y_pred)

            # persist model if model-accuracy is satisfactory
            if self.root_mean_squared_error < threshold_rmse:
                dump(self.model,"model.pkl")
                logging.info("model saved on disk")

            # make submission file
            if make_submission:
                self.get_final_submission_file(x_val,scaler_y)

            # generating plots
            if plot:
                # making a combined dataframe
                df = pd.DataFrame({"row_num": y_test["row_num"].values, "true": y_test["distributed_hits"].values,
                                   "pred": y_pred[:, 0]})
                # aggregating scores for each row number
                df = df.groupby("row_num").sum()
                self.get_plots(df["true"],df["pred"])

            print("RMSE on test dataset : {}".format(self.root_mean_squared_error))
            logging.info("Pipeline ended!")
            return None
        except AssertionError as error:
            logging.error("Path cannot be null")



if __name__ == "__main__":
    """
    A small construct has been made below to make the code executable from the command line interface as well.
    """
    try:
        # suppressing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # creating a parser to manage inputs from terminal
            parser = argparse.ArgumentParser(argument_default=argparse.PARSER)
            parser.add_argument('path', type=str, help='path (including filename) to the dataset on the disk relative to this file e.g. "data/data.csv"')
            parser.add_argument('n_rows', type=int, help='the number of rows to read from the disk e.g 100')
            parser.add_argument('make_submission', type=bool, help='to dump the output or not e.g. False')
            args = parser.parse_args()
            if args.n_rows <= 0:
                logging.warning("number of rows must be > 0")
                n_rows = 100
            else:
                n_rows = int(args.n_rows)
            if args.path == "":
                logging.warning("path not given searching data in default location under data/data.csv")
                filepath = "data/data.csv"
            else:
                filepath = args.path
            make_submission = args.make_submission
            HitsRegression().main(path=filepath, n_rows=n_rows, make_submission=make_submission)
    except:
        # suppressing warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # in case the program is not run from terminal or one-or-more inputs are not given
            logging.warning("no inputs given running with default values")
            filepath = "data/data.csv"
            n_rows = None
            make_submission = True
            HitsRegression().main(path=filepath, n_rows=n_rows, make_submission=make_submission)





