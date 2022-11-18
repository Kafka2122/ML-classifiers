import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class data_classification():

    def __init__(self,clf_opt='svm'):
        self.clf_opt=clf_opt

        # Selection of classifiers

    def classification_pipeline(self):

        # creating the different classifiers

        # Support Vector Machine

        if self.clf_opt=='svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)
            # clf_parameters = {
            #     'clf__C':(0.1,1,100),
            # }


        # Logistic Regression

        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='newton-cg')
            # clf_parameters = {
            # 'clf__random_state':(0,10),
            #     }

        # KNN classifier

        elif self.clf_opt == 'knn':
            print('\n\t### Training K Nearest Neighbour Classifier ### \n')
            clf = KNeighborsClassifier(n_neighbors=16)

        # Bernoulli Naive Bayes

        elif self.clf_opt == 'nb':
            print('\n\t### Training Bernoulli Naive Bayes Classifier ### \n')

            clf = BernoulliNB()


        else:
            print('Select a valid classifier \n')
            sys.exit(0)
        return clf

    def get_data(self):

        # importing the training data file into a dataframe using pandas

        df =  pd.read_csv(r'training_data.csv', header=None) # change path if required

        # importing the training data labels

        labels = pd.read_csv(r'training_data_class_labels.csv',header=None)

        # Joining the training data with class labels

        training_data = pd.concat([df,labels],axis=1,join='inner')

        # Renaming the columns of our data

        colnames = ['column1', 'column2','labels'] # defining a column string

        training_data.columns = colnames # adding the column names

        # importing the test data

        test_data = pd.read_csv(r'test_data.csv', header=None)

        # returning data

        return df,test_data,labels,training_data

    # plotting the data

    def plot(self):

        # for plotting the training data

        df,test_data,labels,training_data = self.get_data()

        plt.scatter(training_data['column1'][training_data.labels == 1],
                    training_data['column2'][training_data.labels == 1],
                    s= 4,
                    color='red',
                    label='1')

        # plotting all the data with labels 0

        plt.scatter(training_data['column1'][training_data.labels == 0],
                    training_data['column2'][training_data.labels == 0],
                    s=4,
                    color='blue',
                    label='0')
        plt.legend()
        plt.show()



    def classification(self,clf_opt):

        # for classification of test data and csv file output

        trn_data, tst_data, trn_cat, whole_data = self.get_data()
        clf = self.classification_pipeline()
        clf.fit(trn_data,trn_cat)
        predicted = clf.predict(tst_data)
        self.output_file(predicted,clf_opt)

    def score(self):

        # spliiting the training data and training labels
        # we apply the classifiers on train data and test data obtained from splitting

        trn_data,tst_data, trn_cat, whole_data = self.get_data()
        x_train, x_test, y_train, y_test = train_test_split(trn_data,trn_cat,test_size=0.20,random_state=10)
        clf = self.classification_pipeline()
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)
        return print(classification_report(y_test,y_predict))

    def output_file(self,predicted,clf_opt):

        # creating a csv output file

        with open(clf_opt+"result.csv", 'w') as f:
            for i in predicted:
                f.write(str(i)+'\n')

