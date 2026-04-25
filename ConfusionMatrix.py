#Creates a Confusion Matrix
#Carl Dennis SI:007968429
#Dominic Mendoza SI:012264773

import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from config import DataDir

#data_root = r"C:\Users\carld\OneDrive\Documents\School\EECE 565\Python\MRICancerClassification\cleaned"
data_root = r"C:\Users\Dominic Mendoza\Documents\EECE_565_Project\MRICancerClassification\cleaned"
train_dir = os.path.join(data_root, "Training")
test_dir = os.path.join(data_root, "Testing")

#Example that was taken from
#https://www.w3schools.com/python/python_ml_confusion_matrix.asp

actual = np.random.binomial(1,.9,size = 1000)
predicted = np.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

#Accuracy
Accuracy = metrics.accuracy_score(actual, predicted)
print(Accuracy)

#Specificity
Specificity = metrics.recall_score(actual, predicted, pos_label=0)
print(Specificity)

#Another Useful Link
#https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/

