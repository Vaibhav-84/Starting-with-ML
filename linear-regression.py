import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes = datasets.load_diabetes()
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'] 

# print(diabetes.DESCR)

diabetes_x = diabetes.data[:, np.newaxis, 2]
# diabetes_x = diabetes.data


# print(diabetes_x)


diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]


diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predicted =  model.predict(diabetes_x_test)

print("MSE is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

print("w :", model.coef_)
print("c :", model.intercept_)

plt.scatter(diabetes_x_test, diabetes_y_test)
plt.plot(diabetes_x_test, diabetes_y_predicted)

plt.show()


################# One feature  #############33
# MSE is:  3035.0601152912695
# w : [941.43097333]
# c : 153.39713623331698

################# All feature  #############33
# MSE is:  1826.5364191345425
# w : [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# c : 153.05827988224112