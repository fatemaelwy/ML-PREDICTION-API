from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np

import pickle
from joblib import dump, load
import json

import math

def convert_data(json_array):
    data_json = json.loads(json_array)
    data = pd.DataFrame(data_json)
    # Define group size
    group_size = 20

    # take the first 20 readings
    groups = data.iloc[:group_size]

    # Calculate the mean of each group for gyro_x, gyro_y, and gyro_z
    raw_gyro = groups.mean(numeric_only=True)[['gyro_x', 'gyro_y', 'gyro_z']]

    # Convert raw gyro readings to radians per second
    raw_gyro_rad = raw_gyro.apply(lambda x: x * math.pi / 180)

    # Integrate the gyro readings over time to get the change in angle for each axis
    delta_theta = raw_gyro_rad.cumsum()

    # Convert the change in angle for each axis to degrees
    delta_theta_deg = delta_theta.apply(lambda x: x * 180 / math.pi)

    # Calculate the orientation, roll, and pitch for each group of 20 readings
    orientation =  math.atan2(delta_theta_deg['gyro_y'], delta_theta_deg['gyro_x'])
    roll = math.atan2(delta_theta_deg['gyro_x'], delta_theta_deg['gyro_z'])
    pitch = math.atan2(delta_theta_deg['gyro_y'], delta_theta_deg['gyro_z'])

    # Calculate the difference in acceleration for each axis
    data['delta_gyro_x'] = data['gyro_x'].diff()
    data['delta_gyro_y'] = data['gyro_y'].diff()
    data['delta_gyro_z'] = data['gyro_z'].diff()

    # Calculate the time elapsed between each reading
    data['delta_time'] = data['time'].diff()

    # Calculate the jerk for each axis by dividing the difference in acceleration by the time elapsed
    data['gyro_jerk_x'] = data['delta_gyro_x'] / data['delta_time']
    data['gyro_jerk_y'] = data['delta_gyro_y'] / data['delta_time']
    data['gyro_jerk_z'] = data['delta_gyro_z'] / data['delta_time']

    # Group the data by every 20 readings and calculate the mean jerk for each group
    jerk_means = data.groupby(data.index // 20).mean()[['gyro_jerk_x', 'gyro_jerk_y', 'gyro_jerk_z']]

    [gyro_jerk_x, gyro_jerk_y, gyro_jerk_z]=[jerk_means['gyro_jerk_x'], jerk_means['gyro_jerk_y'], jerk_means['gyro_jerk_z']]

    overall_gyro_jerk = np.sqrt(gyro_jerk_x**2 + gyro_jerk_y**2 + gyro_jerk_z**2)
    
    jerk_means['overall_gyro_jerk'] = overall_gyro_jerk[0]
    jerk_means['gyro_mean_jerk']=jerk_means['overall_gyro_jerk']


    # Calculate the difference in acceleration for each axis
    data['delta_acc_x'] = data['acc_x'].diff()
    data['delta_acc_y'] = data['acc_y'].diff()
    data['delta_acc_z'] = data['acc_z'].diff()

    # Calculate the time elapsed between each reading
    data['delta_time'] = data['time'].diff()

    # Calculate the jerk for each axis by dividing the difference in acceleration by the time elapsed
    data['acc_jerk_x'] = data['delta_acc_x'] / data['delta_time']
    data['acc_jerk_y'] = data['delta_acc_y'] / data['delta_time']
    data['acc_jerk_z'] = data['delta_acc_z'] / data['delta_time']

    # Group the data by every 20 readings and calculate the mean jerk for each group
    acc_jerk_means = data.groupby(data.index // 20).mean()[['acc_jerk_x', 'acc_jerk_y', 'acc_jerk_z']]

    [acc_jerk_x, acc_jerk_y, acc_jerk_z]=[acc_jerk_means['acc_jerk_x'], acc_jerk_means['acc_jerk_y'], acc_jerk_means['acc_jerk_z']]

    overall_acc_jerk = np.sqrt(acc_jerk_x**2 + acc_jerk_y**2 + acc_jerk_z**2)

    acc_jerk_means['overall_acc_jerk']=overall_acc_jerk

    mean_acc_jerk=(acc_jerk_x+ acc_jerk_y+acc_jerk_z)/3

    jerk_means['acc_jerk_x']=acc_jerk_x
    jerk_means['acc_jerk_y']=acc_jerk_y
    jerk_means['acc_jerk_z']=acc_jerk_z
    jerk_means['mean_acc_jerk']=mean_acc_jerk
    jerk_means['overall_acc_jerk']=overall_acc_jerk



    # drop the unnecessary columns
    df = pd.DataFrame(data_json)
    df = df.drop(["time"], axis=1)
    df = df.drop(["distance"], axis=1)
    df = df.drop(["accident"], axis=1)

    # split the data into groups of 20 readings
    groups = df.groupby(np.arange(len(df)) // 20)

    df= pd.DataFrame(data_json)

    df=df.drop(["time"], axis = 1)
    df=df.drop(["distance"], axis = 1)
    df=df.drop(["accident"], axis = 1)

    # split the data into groups of 20 readings
    groups = df.groupby(np.arange(len(df))//20)




   # calculate the mean, standard deviation, median, min, max, and coefficient of variation for each group
    results = groups.agg(['mean', 'std', 'median', 'min', 'max', 'sem'])

    results.columns = results.columns.map(lambda x: f'{x[0]}_{x[1]}')


    pre_processed_df = pd.concat([jerk_means, results], axis=1)

    pre_processed_df['orientation']=orientation
    pre_processed_df['pitch']=pitch
    pre_processed_df['roll']=roll

    json_str = pre_processed_df.to_json(orient='records')

    return json_str





clf = load('decision_tree.joblib')


app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def predict():
    # Get the input data from the JSON payload
    data = request.get_json()

    
    data=convert_data(json.dumps(data))
    

    data=data[1:-1]
    data=json.loads(data)
  
    input_data = np.array([data['gyro_jerk_x'], data['gyro_jerk_y'], data['gyro_jerk_z'],
                           data['overall_gyro_jerk'], data['gyro_mean_jerk'], data['acc_jerk_x'],
                           data['acc_jerk_y'], data['acc_jerk_z'], data['mean_acc_jerk'],
                           data['overall_acc_jerk'], data['acc_x_mean'], data['acc_x_std'],
                           data['acc_x_median'], data['acc_x_min'], data['acc_x_max'], data['acc_x_sem'],
                           data['acc_y_mean'], data['acc_y_std'], data['acc_y_median'], data['acc_y_min'],
                           data['acc_y_max'], data['acc_y_sem'], data['acc_z_mean'], data['acc_z_std'],
                           data['acc_z_median'], data['acc_z_min'], data['acc_z_max'], data['acc_z_sem'],
                           data['gyro_x_mean'], data['gyro_x_std'], data['gyro_x_median'], data['gyro_x_min'],
                           data['gyro_x_max'], data['gyro_x_sem'], data['gyro_y_mean'], data['gyro_y_std'],
                           data['gyro_y_median'], data['gyro_y_min'], data['gyro_y_max'], data['gyro_y_sem'],
                           data['gyro_z_mean'], data['gyro_z_std'], data['gyro_z_median'], data['gyro_z_min'],
                           data['gyro_z_max'], data['gyro_z_sem'], data['orientation'], data['pitch'],
                           data['roll']])
    
    

    # Make a prediction using the trained classifier
    prediction = clf.predict(input_data.reshape(1, -1))
    
    
    # Convert the prediction to a JSON response
    response = {'maneuv_type': prediction}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

    