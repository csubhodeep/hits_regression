# hits_regression

Build a model that predicts the number of hits per session.



# Data description

The data in .csv format containing ~900k records must be kept in the `data` directory. 

The columns should be understood as follows:
+ `row_num`: a number uniquely identifying each row.
+ `locale`: the platform of the session.
+ `day_of_week`: Mon-Fri, the day of the week of the session.
+ `hour_of_day`: 00-23, the hour of the day of the session.
+ `agent_id`: the device used for the session.
+ `entry_page`: describes the landing page of the session.
+ `path_id_set`: shows all the locations that were visited during the session.
+ `traffic_type`: indicates the channel the user cane through eg. search engine, email, ...
+ `session_duration`: the duration in seconds of the session.
+ `hits`: the number of interactions with the trivago page during the session.


# Objective

Predict the number of hits corresponding to the rows where the column `hits` has missing values. Predictions are evaluated by the metric of root mean square error.

# Solution

The code `hits_regression.py` contains the main class that is used for training the model. The code can be executed from the CLI as well. By default the `make_submission` argument of the `main` method of the class is set to `True`, which will create a separate CSV file containing 2 columns - `row_num` and `hits` for the all those rows where predictions are to be made. 
