import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
scaler = StandardScaler()

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

KNN = False
Random_Forest = False
Decision_Trees = False
Logistic_Regression = False 
Neural_Networks = False


################################################################
#                     Plotting Functions                       #
################################################################

def plot_grid_search_1d(cv_results, grid_param, name_param, text):
    plt.plot(grid_param, cv_results['mean_test_score'], '-o')
    plt.xlim(1,len(grid_param)-1)
    plt.xticks(grid_param)
    plt.xlabel(name_param, fontsize=10)
    plt.ylabel('CV Average Validation Accuracy', fontsize=10)
    plt.title("Grid Search Scores", fontsize=12, fontweight='bold')
    plt.text(x=0, y=-0.05, s=text, fontsize=10, transform=plt.gca().transAxes)
    plt.grid('on')
    plt.show()

def plot_grid_search_2d(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores", fontsize=12, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=10)
    ax.set_ylabel('CV Average Validation Accuracy', fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid('on')

# Preperation
data_ori = pd.read_csv("ClassifyProducts.csv")
data = data_ori.drop(["id"] , axis = 1)
sampled_data_list = []

report = pd.DataFrame(columns=["Model",
                                "Sampling Method", 
                                "Test Size",
                                "Training Acc.",
                                "Test Acc.",
                                "Cross Valid Acc.",
                                "Standard Deviation",
                                "Best Parameter",
                                "Time (minutes)"])


################################################################
#                        Data Sampling                         #
################################################################


# Downsampling (Static)
###############################################################################
min_count = data["target"].value_counts().min()
data_downsampled = pd.DataFrame()

for i in data["target"].unique():
    class_data_downsampling = data[data["target"] == i]
    
    current_count = class_data_downsampling.shape[0]
    
    if current_count > min_count:
        downsampled_class = resample(class_data_downsampling,
                                     replace=False,
                                     n_samples=min_count,
                                     random_state=0)
    else:
        downsampled_class = class_data_downsampling
       
    data_downsampled = pd.concat([data_downsampled, downsampled_class])
sampled_data_list.append(data_downsampled)
    

# Upsampling durch Duplizierung
###############################################################################
max_count = data["target"].value_counts().max()
upsampled_data_temp = []

for i in data["target"].unique():
    class_data = data[data["target"] == i]

    upsampled_class = resample(class_data,
                               replace=True,
                               n_samples=max_count,
                               random_state=0)

    upsampled_data_temp.append(upsampled_class)

data_upsampled = pd.concat(upsampled_data_temp)     
sampled_data_list.append(data_upsampled)



# Combination Sampling
###############################################################################
from sklearn.utils import resample
avg_count = int(data["target"].value_counts().mean())

up_data = []
down_data = []
for i in data["target"].unique():
    class_data = data[data["target"] == i]

    if len(class_data) > avg_count:

        down_class = resample(class_data,
                                     replace=False,
                                     n_samples=avg_count,
                                     random_state=0)
        down_data.append(down_class)
        
    else:
        up_class = resample(class_data,
                                   replace=True,
                                   n_samples=avg_count,
                                   random_state=0)
        up_data.append(up_class)

combination_data = pd.concat(up_data + down_data)
sampled_data_list.append(combination_data)


# SMOTE
###############################################################################
###############################################################################

sampling_model_names = ["Downsampling" , "Upsampling" , "Combination"]
test_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

start_time = time.time()                             

for test_size in test_sizes:
    for model_name, model_data in zip(sampling_model_names, sampled_data_list):
        

        ################################################################
        #                       Scaling the Data                       #
        ################################################################
        
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder()
        target_array = ohe.fit_transform(model_data[["target"]]).toarray()
        target_labels = ohe.categories_
        target_labels = np.array(target_labels).ravel()
        
        X = model_data.drop(["target"] , axis = 1)
        Y = pd.DataFrame(target_array, columns=target_labels).drop("Class_1" , axis = 1)
               
        # Normalize
        from sklearn import preprocessing
        nscaler = preprocessing.MinMaxScaler()
        X.iloc[:,:] = nscaler.fit_transform(X.iloc[:,:])  # Trick to keep the dataframe
        
        ################################################################
        #               Train Test Split + Cross Valid.                #
        ################################################################
        
        from sklearn.model_selection import train_test_split
        
        # Splitting Up
        X_train , X_rem, Y_train, Y_rem = train_test_split(X , Y , 
                                                             stratify = Y,
                                                             test_size=test_size, 
                                                             random_state=0)
        
        # Splitting Up
        X_valid , X_test, Y_valid, Y_test = train_test_split(X_rem , Y_rem, 
                                                             stratify = Y_rem,
                                                             test_size=0.5, 
                                                             random_state=0)
            
        
        ################################################################
        #                             KNN                              #
        ################################################################
        if KNN:
            print("Start: KNN")
            
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import GridSearchCV
            from sklearn.metrics import accuracy_score, confusion_matrix
            from sklearn.metrics import multilabel_confusion_matrix
            
            knnmodel = KNeighborsClassifier()
            param_grid = {'n_neighbors': range(1, 31, 2)}
            
            CV_knnmodel = GridSearchCV(estimator=knnmodel, param_grid=param_grid, cv=10)
            CV_knnmodel.fit(X_train, Y_train)
            
            print("Best parameter:" , CV_knnmodel.best_params_)
            
            # Use the best parameters
            knnmodel = knnmodel.set_params(**CV_knnmodel.best_params_)
            knnmodel.fit(X_train, Y_train)
            Y_test_pred = knnmodel.predict(X_test)
            accte = accuracy_score(Y_test, Y_test_pred)
    
           
            # Plotting
            plot_model = CV_knnmodel.cv_results_
            param_name = list(param_grid.keys())[0]
            param_values = param_grid[param_name]
            text = "Sampling Model:", model_name, " , Training Size:", 1 - test_size
            plot_grid_search_1d(plot_model, param_values, param_name, text)
            
            # Cross Validation: Model has already been trained
            Y_valid_pred = knnmodel.predict(X_valid)
            accvalid = accuracy_score(Y_valid, Y_valid_pred)
            print("Final Score after validation process:" , accvalid)
            
            end_time = time.time()
            mins = round((end_time - start_time) / 60 , 2)
            start_time = time.time()
            print(model_name, test_size, "done! (" + str(mins) + "mins)")
            
            # Report
            report.loc[len(report)] = ['k-NN (grid)',
                                      model_name,
                                      str(test_size),
                                      CV_knnmodel.cv_results_['mean_test_score'][CV_knnmodel.best_index_],
                                      accte,
                                      accvalid,
                                      CV_knnmodel.cv_results_['std_test_score'][CV_knnmodel.best_index_],
                                      CV_knnmodel.best_params_['n_neighbors'],
                                      mins]
            
        
            report.to_excel("report_knn.xlsx")
        
        
        
        
        
        
        
        
        ################################################################
        #                       Random Forest                          #
        ################################################################
        if Random_Forest:
            print("Random Forest startet")
            
            start_time = time.time()
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import GridSearchCV
            rfmodel = RandomForestClassifier(random_state=0)
            param_grid = { 
                'max_depth': [ 4,  5,  6,  7,  8],       #as there are 5x5 values we have 25 cross validation grid
                'n_estimators': [ 10,  50,  100, 150, 200]
            }
            CV_rfmodel = GridSearchCV(estimator=rfmodel, param_grid=param_grid, cv=10)
            CV_rfmodel.fit(X_train, Y_train)
            print(CV_rfmodel.best_params_)
            
            # Use the best parameters
            rfmodel = rfmodel.set_params(**CV_rfmodel.best_params_)
            rfmodel.fit(X_train, Y_train)
            Y_test_pred = rfmodel.predict(X_test)
            accte = accuracy_score(Y_test, Y_test_pred)
            
            # Plotting
            plot_model = CV_rfmodel.cv_results_
            param_name1 = list(param_grid.keys())[0]
            param_name2 = list(param_grid.keys())[1]
            param1_values = param_grid[param_name1]
            param2_values = param_grid[param_name2]
            plot_grid_search_2d(plot_model, param1_values, param2_values, param_name1, param_name2)
            
            # Cross Validation: Model has already been trained
            Y_valid_pred = rfmodel.predict(X_valid)
            accvalid = accuracy_score(Y_valid, Y_valid_pred)
            print("Final Score after validation process:" , accvalid)
    
    
    
            end_time = time.time()
            mins = round((end_time - start_time) / 60, 2)
            start_time = time.time()
            
            rf_best = "Max Depth:", CV_rfmodel.best_params_["max_depth"], " , n_estimators:", CV_rfmodel.best_params_["n_estimators"]
            report.loc[len(report)] = ["Random Forest (grid)",
                                      model_name,
                                      str(test_size),
                                      CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_],
                                      accte,
                                      accvalid,
                                      CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_],
                                      rf_best,
                                      mins]
            print(report.loc[len(report)-1])
        
        
        
        
        
        
        
        
        
        
        
        
        
        ################################################################
        #                      Decision Trees                          #
        ################################################################     
        if Decision_Trees:
            
            from sklearn.tree import DecisionTreeClassifier
            etmodel = DecisionTreeClassifier(random_state=0)
            from sklearn.model_selection import GridSearchCV
            param_grid = { 
                'criterion': ['entropy', 'gini'],
                'max_depth': range(1, 21)
            }
            CV_etmodel = GridSearchCV(estimator=etmodel, param_grid=param_grid, cv=10)
            CV_etmodel.fit(X_train, Y_train)
            print(CV_etmodel.best_params_)
            
            # Use best parameters
            etmodel = etmodel.set_params(**CV_etmodel.best_params_)
            etmodel.fit(X_train, Y_train)
            Y_test_pred = etmodel.predict(X_test)
            accte = accuracy_score(Y_test, Y_test_pred)
            
            # Plotting
            plot_model = CV_etmodel.cv_results_
            param_name1 = list(param_grid.keys())[1]
            param_name2 = list(param_grid.keys())[0]
            param1_values = param_grid[param_name1]
            param2_values = param_grid[param_name2]
            plot_grid_search_2d(plot_model, param1_values, param2_values, param_name1, param_name2)
            
            # Cross Validation: Model has already been trained
            Y_valid_pred = etmodel.predict(X_valid)
            accvalid = accuracy_score(Y_valid, Y_valid_pred)
            print("Final Score after validation process:" , accvalid)
            
            # Report
            dt_best = "Criterion", CV_etmodel.best_params_["criterion"], " , Max Depth:", CV_etmodel.best_params_["max_depth"]
            report.loc[len(report)] = ["Decision Tree (grid)",
                                      model_name,
                                      str(test_size),
                                      CV_etmodel.cv_results_['mean_test_score'][CV_etmodel.best_index_],
                                      accte,
                                      accvalid,
                                      CV_etmodel.cv_results_['std_test_score'][CV_etmodel.best_index_],
                                      dt_best]
            print(report.loc[len(report)-1])
        
        
        
        
        
        
        
        
        
        
        
        
        ################################################################
        #                     Logistic Regression                      #
        ################################################################
        if Logistic_Regression:
            
        
            # Convert Y 
            Y_train2 =(Y_train.eq(1)).idxmax(1).str.extract('(\d+)').astype(int)
            Y_test2 = (Y_test.eq(1)).idxmax(1).str.extract('(\d+)').astype(int)
            Y_valid2 = (Y_valid.eq(1)).idxmax(1).str.extract('(\d+)').astype(int)
            
            from sklearn.linear_model import LogisticRegression
            lrmodel = LogisticRegression()
            from sklearn.model_selection import cross_val_score
            accuracies = cross_val_score(lrmodel, X_train, Y_train2, scoring='accuracy', cv = 10)
            lrmodel.fit(X_train, Y_train2)
            Y_test_pred = lrmodel.predict(X_test)
            accte = accuracy_score(Y_test2, Y_test_pred)
            
            # Cross Validation: Model has already been trained
            Y_valid_pred = lrmodel.predict(X_valid)
            accte = accuracy_score(Y_valid2, Y_valid_pred)
            print("Final Score after validation process:" , accte)
            
            # Report
            report.loc[len(report)] = ["Logistic Regression",
                                      model_name,
                                      str(test_size),
                                      accuracies.mean(),
                                      accte,
                                      accvalid,
                                      accuracies.std(),
                                      0]
            print(report.loc[len(report)-1])
            
            
            
            
            
            
            
            
        
        ################################################################
        #                       Neural Networks                        #
        ################################################################
        if Neural_Networks:
        
        
            from sklearn.neural_network import MLPClassifier
            nnetmodel = MLPClassifier(solver='lbfgs', random_state=0, max_iter=1000) #max_iter standard ist 200
            from sklearn.model_selection import GridSearchCV
            param_grid = { 
                'hidden_layer_sizes': [(3,), (5,), (8,), (10,), (13,), (15,), (17,), (19,)]
            }
            CV_nnetmodel = GridSearchCV(estimator=nnetmodel, param_grid=param_grid, cv=10)
            CV_nnetmodel.fit(X_train, Y_train)
            print(CV_nnetmodel.best_params_)
            #use the best parameters
            
            nnetmodel = nnetmodel.set_params(**CV_nnetmodel.best_params_)
            nnetmodel.fit(X_train, Y_train)
            Y_test_pred = nnetmodel.predict(X_test)
            accte = accuracy_score(Y_test, Y_test_pred)
            
            # Plotting
            plot_model = CV_nnetmodel.cv_results_
            param_name = list(param_grid.keys())[0]
            param_values = np.array(param_grid[param_name])[:,0]   # must be converted her because of list of tuples
            plot_grid_search_1d(plot_model, param_values, param_name)
            
            
            # Cross Validation: Model has already been trained
            Y_valid_pred = nnetmodel.predict(X_valid)
            accvalid = accuracy_score(Y_valid, Y_valid_pred)
            print("Final Score after validation process:" , accvalid)
            
            # Report
            nn_best = "Hidden Layer Size:" , CV_nnetmodel.best_params_["hidden_layer_sizes"]
            report.loc[len(report)] = ["Neural Network (grid)",
                                      model_name,
                                      str(test_size),
                                      CV_nnetmodel.cv_results_['mean_test_score'][CV_nnetmodel.best_index_],
                                      accte,
                                      accvalid,
                                      CV_nnetmodel.cv_results_['std_test_score'][CV_nnetmodel.best_index_]]
            print(report.loc[len(report)-1])
            

print(report)
report.to_excel("report.xlsx")







