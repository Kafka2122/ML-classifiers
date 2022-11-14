from data_class_20260 import data_classification

# for choosing different classifiers change the value of clf_opt

# for SVM type clf_opt ='svm'
# for KNN type clf_opt ='knn'
# for naive bayes type clf_opt ='nb'
# for logistic regression type clf_opt ='lr'

clf_opt = 'nb'

clf=data_classification(clf_opt)


clf.plot() # for plotting
clf.classification(clf_opt) # for classification of test data and csv file output
clf.score() # for classification report

# getting the best score with knn

