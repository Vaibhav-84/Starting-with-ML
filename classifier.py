from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# load datasets 
iris = datasets.load_iris()

# printing description and features            
features = iris.data
labels = iris.target
print(features[0], labels[0])

# Training the classifier           
clf = KNeighborsClassifier()

clf.fit(features, labels)

preds = clf.predict([[53.1, 34.5, 1.4 ,0.2]])
print(preds)