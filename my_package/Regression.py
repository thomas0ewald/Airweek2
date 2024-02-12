import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, linear_model
import pandas as pd
import re
import ast
import pickle
import pandas as pd

class Regression: 
    """ Model for prediction of Distance based on detected people keypoints"""
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def train(self, x):
        """ Predict the distance (y) based on the keypoints (x)"""
        model = linear_model.LinearRegression()

        # train for 30 generations
        best = 0
        for _ in range(30):
            x_train, x_test,y_train, y_test = model_selection.train_test_split(self._x, self._y, test_size=0.1)

            # train the model
            model.fit(x_train, y_train)
            # get accuracy with the actual measurements
            accuracy = model.score(x_test, y_test)
            predictions = model.predict(x_test)

            print(f"score {accuracy}")

            if accuracy > best:
                best = accuracy
            # save model in pickle file
            with open("distance_predictions.pickle", "wb") as f:
                pickle.dump(model, f)

        # load from existing model
        pickle_in = open("distance_predictions.pickle", "rb")
        model = pickle.load(pickle_in)


        # for x in range(len(predictions)):
        #     print(f"predicted label: {predictions[x]}, \n attributes: {x_test}, \n actual label: {y_test[x]}")

        # Coefficients und Intercept ausgeben
        print("Steigung (slope):", model.coef_[0])
        print("Achsenabschnitt (intercept):", model.intercept_)


#-----
def transform_distance_and_intensity(data):
    """ Prepare data for training
        Data from sensors is passed as string: [[8.  9.  7.  ], [0.  7.  1.]]
        and is converted to valid numpy arrays containing floats: [[8,  9,  7], [0,  7,  1]]"""

    # replace points with comma: 8.  9.  7.  with 8, 9, 7
    regex_pattern = r'(?<=\d)\s+(?=\d)'
    string_with_commas = [[re.sub(regex_pattern, ', ', line) for line in data]]

    regex_pattern_replace_zeros = r'(\d\.\s+)'
    cret = [[re.sub(regex_pattern_replace_zeros, r'\1, ', number) for line in string_with_commas for number in line]]

    converted_distance_intensity_list = []

    for all in cret:
        for line in all:
            # get only the numbers from huge sensor string
            numbers_str_list = line.strip('[]').split(', ')
            # convert to float
            numbers_float_list = [float(number) for number in numbers_str_list if number != '' ]
            converted_distance_intensity_list.append(numbers_float_list)

    return converted_distance_intensity_list

def convert_to_float(lst):
    for i, val in enumerate(lst):
        if isinstance(val, list):
            convert_to_float(val)
        else:
            lst[i] = float(val)
    return lst

def transform_keypoints(data):
    regex_pattern = r'(?<=\d)\s+(?=\d)'
    ret = [[re.sub(regex_pattern, ', ', line) for line in data]]
    ccret = [[re.sub(r'\]\s+\[', '], [', tmp) for all in ret for tmp in all]]

    converted_keypoints_list = []

    for all in ccret:
        for line in all:
                nested_list = ast.literal_eval(line)
                converted_keypoints_list.append(convert_to_float(nested_list))

    return converted_keypoints_list



# read recorded data form saved .csv file
csv_file_path = 'data.csv'
df = pd.read_csv(csv_file_path)

df.replace(to_replace='\n', value='', regex=True, inplace=True)

distances = transform_distance_and_intensity(df.Distance)
intensities = transform_distance_and_intensity(df.Intensity)
keypoints = transform_keypoints(df.Keypoint)

keypoints_train_x = []
keypoints_train_y = []
keypoints_train_z = []


# return value from Keypoints:
# "[[[1.18184509e+02 1.08358841e+02 6.85976148e-02]
#   [...            ...             ...          ]
#   [1.14462128e+02 1.31671341e+02 3.28823403e-02]]

#  [[4.24308136e+02 1.15316460e+02 3.49054188e-02]
#   [4.23627899e+02 1.16665146e+02 4.47378010e-02]
#   [...            ...             ...          ]
#   [4.19546295e+02 1.29140427e+02 1.25035867e-02]]

for keypoints in all:
    for keypoint in keypoints:
        for xyz in keypoint:
            keypoints_train_x.append(xyz[0])
            keypoints_train_y.append(xyz[1])
            keypoints_train_z.append(xyz[2])



# loop through distances - ESTIMATION: every 20 distance measurements belong to a keypoint
def get_measured_sensor_distance_for_keypoints():
    splitted_distances = []

    for element in distances:
        print(len(element))
        for i in range(0, len(element), 20):
            if i == 0:
                splitted_distances.append(np.array(element[:i]))
            else:
                splitted_distances.append(np.array(element[i-20:i]))

    return splitted_distances
    # print(len(splitted_distances))


splitted_distances = get_measured_sensor_distance_for_keypoints()

# get non empty arrays
non_empty_arrays = [arr for arr in splitted_distances if len(arr) > 0]
print(len(non_empty_arrays))
print(len(keypoints_train_x))
print('-------------------------')
# print(splitted_distances[1])
print('-------------------------')

# dictionary of lists 
print(len(keypoints_train_x[:2069]))
print(len(distances))
dict = {'x': keypoints_train_x, 'y': keypoints_train_y, 'z': keypoints_train_z} 
    
df = pd.DataFrame(dict)
print(df.head(5))


regression_train = Regression(keypoints[:len(non_empty_arrays)], non_empty_arrays)
regression_train.train(df[:len(non_empty_arrays)])

