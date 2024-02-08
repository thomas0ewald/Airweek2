import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, linear_model
import pandas as pd
import re
import ast

class Regression: 
    def __init__(self, X, Y):
        self._x = X
        self._y = Y

    def train(self):
        # print(self.X)
        # x_axis_keypoints = self.X[:,:,0]
        # print(x_axis_keypoints[:3])
        print("------------------------------")
        # x_axis_distance = self.Y[:,0]
        # print(x_axis_distance[:3])

        #Linear Regression Modell initialisieren und anpassen
        # model = LinearRegression()

        # Umwandeln der verschachtelten Liste in ein NumPy-Array
        # array = np.array(self._x)
        # print(self._x)
        # test = np.array(self._x)
        # test2 = np.array(self._y)
        # print(array)
        # print(test2)
        # model.fit(test, test2)

        # # Vorhersagen treffen
        # # X_new = np.array([[6]])
        # prediction = model.predict(self._x)
        # print("Vorhersage für X_new:", prediction)

        # # Coefficients und Intercept ausgeben
        # print("Steigung (slope):", model.coef_[0])
        # print("Achsenabschnitt (intercept):", model.intercept_)

    def train_x(self, x):
        model = linear_model.LinearRegression()
        print(self._y[0][18])

        x_train, x_test,y_train, y_test = model_selection.train_test_split(x, self._y, test_size=0.1)

        print(x_train)
        print(y_train)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        predictions = model.predict(x_test)

        print(f"score {accuracy}")

        for x in range(len(predictions)):
            print(f"predicted label: {predictions[x]}, \n attributes: {x_test}, \n actual label: {y_test[x]}")

        # Coefficients und Intercept ausgeben
        print("Steigung (slope):", model.coef_[0])
        print("Achsenabschnitt (intercept):", model.intercept_)
        
    
    def predict_x(self, x):
        model = linear_model.LinearRegression()

        x_train, x_test,y_train, y_test = model_selection.train_test_split(x, self._y, test_size=0.1)
        print(y_train)

        model.fit(y_train, x_train)
        accuracy = model.score(y_test, x_test)
        predictions = model.predict(y_test)

        for x in range(len(predictions)):
            print(f"predicted label: {predictions[x]}, \n attributes: {y_test[x]}, \n actual label: {x_test[x]}")

        # Coefficients und Intercept ausgeben
        print("Steigung (slope):", model.coef_[0])
        print("Achsenabschnitt (intercept):", model.intercept_)
        print(f"score {accuracy}")


#-----
def transform_distance_and_intensity(data):
    regex_pattern = r'(?<=\d)\s+(?=\d)'
    ret = [[re.sub(regex_pattern, ', ', line) for line in data]]

    regex_pattern_replace_zeros = r'(\d\.\s+)'
    cret = [[re.sub(regex_pattern_replace_zeros, r'\1, ', number) for line in ret for number in line]]

    bigass_list = []

    for all in cret:
        for line in all:
            numbers_str_list = line.strip('[]').split(', ')
            # tmp = [print(number) for number in numbers_str_list]
            numbers_float_list = [float(number) for number in numbers_str_list if number != '' ]
            bigass_list.append(numbers_float_list)

    return bigass_list

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

    bigass_list = []

    for all in ccret:
        for line in all:
                nested_list = ast.literal_eval(line)
                bigass_list.append(convert_to_float(nested_list))

                # numbers_str_list = line.split('], ')
                # numbers_float_list = [float(number) for number in numbers_str_list]
                # bigass_list.append(numbers_float_list)

    return bigass_list

# def transform_keypoints(data):




#read train data from file
csv_file_path = 'data.csv'

# Lesen der CSV-Datei
df = pd.read_csv(csv_file_path)
# df['Distance'] = df['Distance'].astype(float)


df.replace(to_replace='\n', value='', regex=True, inplace=True)

distances = transform_distance_and_intensity(df.Distance)
intensities = transform_distance_and_intensity(df.Intensity)
keypoints = transform_keypoints(df.Keypoint)

# print(keypoints[0][0][0][0])
keypoints_train_x = []
keypoints_train_y = []
keypoints_train_z = []

for all in keypoints:
    for not_all in all:
        for even_less in not_all:
            keypoints_train_x.append(even_less[0])
            keypoints_train_y.append(even_less[1])
            keypoints_train_z.append(even_less[2])



# importing pandas as pd 
import pandas as pd
  


# loop through distances - ESTIMATION: every 20 distance measurements belong to a keypoint
splitted_distances = []

for element in distances:
    print(len(element))
    for i in range(0, len(element), 20):
        if i == 0:
            splitted_distances.append(np.array(element[:i]))
        else:
            splitted_distances.append(np.array(element[i-20:i]))

    print('----------------')

print(len(splitted_distances))

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
regression_train.train_x(df[:len(non_empty_arrays)])




# df.replace(to_replace='  ', value=',', regex=True, inplace=True)


# print(df.head())





# print([[line for line in ret]])


# # Ersetzen des gefundenen Musters durch "0.," also "0." gefolgt von einem Komma



# print([[line for line in df.Distance]])

# Ausgabe des modifizierten Strings
# print(cret[0][0])


