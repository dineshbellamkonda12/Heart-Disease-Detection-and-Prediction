#!/usr/bin/env python
# coding: utf-8

# # **Importing** **Required Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick
import pickle
import warnings
import os
from matplotlib.collections import PathCollection
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.style import set_palette
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap

# Applying Warning Filters
warnings.filterwarnings('ignore')


# **Loading Dataset**

# In[2]:


HD_data = pd.read_csv("Heart.csv")
HD_data


# In[3]:


print('Total number of rows in Dataset are :', HD_data.shape[0])
print('Total number of columns in dataset are:', HD_data.shape[1])
print('\n')
print('Details of the dataset are given below :')
print('-----------------------------------------')
HD_data.info()


# In[4]:


list_obj =['restecg','slope', 'sex', 'cp', 'thal', 'fbs', 'exang', 'ca']
HD_data[list_obj] = HD_data[list_obj].astype(object)


# # Initial Data Exploration

# In[5]:


# Gender
colors=['Green','Blue', 'Yellow', "Cyan", "Red"]
Head = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target' ]
Title = ['Sex (Gender) Distribution', 'Chest Pain Type Distribution', 'Fasting Blood Sugar Distribution','Resting Electrocardiographic Distribution','Exercise Induced Angina Distribution','Slope of the Peak Exercise Distribution','Number of Major Vessels Distribution','thal Distribution','Heart Diseases Distribution']
Xlabel = ['Gender','Pain Type','Fasting Blood Sugar','Resting Electrocardiographic','Exercise Induced Angina','Slope','Number of Major Vessels','Number of "thal','Heart Disease Status']

labels=[['Female', 'Male'],['Type 0', 'Type 2', 'Type 1', 'Type 3'],['< 120 mg/dl', '> 120 mg/dl'],['1', '0', '2'],['False', 'True'],['2', '1', '0'],['0', '1', '2', '3', '4'],['2', '3', '1', '0'],['True', 'False']]
for i in range (9):
  order = HD_data[Head[i]].value_counts().index
  # Declaring Size for pie chart and histogram
  plt.figure(figsize=(14, 7))
  plt.suptitle(Title[i], fontfamily='Verdana', fontweight='heavy', 
              fontsize='13', color='#100C07')
  # Pie Chart Implementation
  plt.subplot(1, 2, 1)
  plt.title('Pie Chart', fontfamily='verdana', fontweight='heavy', fontsize=13,
            color='#100C07')
  plt.pie(HD_data[Head[i]].value_counts(), labels=labels[i], textprops={'fontsize':13}, colors=colors, pctdistance=0.6,
           wedgeprops=dict(alpha=0.9, edgecolor='#3E3B39'), autopct='%.2f%%')
  cen=plt.Circle((0, 0), 0.43, edgecolor= '#3E3B39', fc='white')
  plt.gcf()
  plt.gca().add_artist(cen)

  # Histogram Implementation
  cntplot = plt.subplot(1, 2, 2)
  plt.title('Histogram', fontsize=13, fontfamily='verdana',color='#100C07', 
            fontweight='heavy')
  ax = sns.countplot(x=Head[i], data=HD_data, palette=colors, order=order,
                    edgecolor='#6D6A6A', alpha=0.87)
  for rect in ax.patches:
      ax.text (rect.get_x()+rect.get_width()/2, 
              rect.get_height()+4.3,rect.get_height(), 
              fontsize=12, horizontalalignment='center', 
              bbox=dict(boxstyle='round', facecolor='none',linewidth=0.28, 
                        edgecolor='#100C07'))

  plt.xlabel(Xlabel[i], fontweight='heavy',color='#3E3B39', fontfamily='verdana', 
             fontsize=10)
  plt.ylabel('Total', color='#3E3B39', fontweight='heavy', fontfamily='verdana', 
              fontsize=10)
  plt.grid(alpha=0.5, axis='y')
  cntplot


# # **Numerical Variable**

# **Descriptive Statistics**

# In[6]:


HD_data.select_dtypes(exclude='object').describe()


# In[7]:


Describe = ['age','trestbps','chol','thalach','oldpeak']
colors=['#FF0000','#8A0030', '#4C0028', "#F38BB2", "#400000"]
Ytitle = ['Age','Resting Blood Pressure','Serum Cholestoral','Maximum Heart Rate','oldpeak']
for i in range(len(Describe)):
  # Declaring Color, Variable and Size of plot
  vab = Describe[i]
  clr = colors[i]
  fig=plt.figure(figsize=(11, 10))

  # Declaring Skewness and Kurtosis
  print('Skewness is:', HD_data[vab].skew(skipna = True, axis = 0))
  print('Kurtosis is:', HD_data[vab].kurt(skipna = True, axis = 0))
  print('\n')

  # Declaring Histogram
  axs1=fig.add_subplot(2, 2, 2)
  plt.title('Plot for Histogram', color='#3E3B39', fontweight='heavy',
            fontfamily='verdana', fontsize=13)
  sns.histplot(data=HD_data,color=clr, kde=True, x=vab)
  plt.xlabel('Total', color='#3E3B39', fontweight='heavy', 
            fontfamily='verdana', fontsize=13)
  plt.ylabel(Ytitle[i], color='#3E3B39', fontweight='heavy', 
             fontfamily='verdana', fontsize=13)

  # Declaring Q-Q Plot
  axs2=fig.add_subplot(2, 2, 4)
  plt.title('Q Q Plot', color='#3E3B39', fontweight='heavy', 
            fontfamily='verdana',fontsize=13)
  qqplot(HD_data[vab], line='45', markerfacecolor=clr, ax=axs2, 
        markeredgecolor=clr, fit=True, alpha=0.5)
  plt.xlabel('Theoritical Quantiles', color='#3E3B39', fontweight='heavy', 
            fontfamily='verdana', fontsize=13)
  plt.ylabel('Sample Quantiles', color='#3E3B39', fontweight='heavy',
            fontfamily='verdana', fontsize=13)

  #Declaring Box Plot
  axs3=fig.add_subplot(1, 2, 1)
  plt.title('Box Plot', color='#3E3B39', fontweight='heavy', fontfamily='verdana', 
            fontsize=13)
  sns.boxplot(data=HD_data, linewidth=1.5,boxprops=dict(alpha=0.7), color=clr, y=vab)
  plt.ylabel(Ytitle[i], color='#3E3B39', fontweight='heavy', fontfamily='verdana', 
             fontsize=13)

  plt.show()


# # **Exploratory Data Analysis**

# **Heart Disease Distribution based on Gender**

# In[8]:


# Bar Chart
ax = pd.crosstab(HD_data.sex, HD_data.target).plot(edgecolor='#6D6A6A', kind='bar',
                                         color=['#FFD7D7', '#F17881'], figsize=(9, 7), 
                                         alpha=0.85)
# Declaring parameters for barchart
for rect in ax.patches:
    ax.text (rect.get_x()+rect.get_width()/2, 
             rect.get_height()+1.3,rect.get_height(), 
             fontsize=12, horizontalalignment='center')

plt.suptitle('Distribution of Heart Disease by Gender',color='#100C07',  
             fontfamily='verdana', x=0.01, y=0.1, fontsize='16', 
              fontweight='heavy', ha='left')
plt.tight_layout(rect=[0.01, 0.10, 0.9, 1.084])
plt.title('In male, the distribution is not imbalanced.The female have almost the same distribution. \n Females have most of the heart diseases when compared to Male.', 
          fontsize='8', fontfamily='verdana', loc='left', color='#3E3B39')
plt.xlabel('Gender (Sex)', color='#3E3B39', fontweight='heavy', 
           fontfamily='verdana')
plt.ylabel('Total Number of People', color='#3E3B39', fontweight='heavy', 
           fontfamily='verdana')
labls = ['False', 'True']
gender_arr = np.array([0, 1])
gender_MF = ['Male', 'Female']
plt.xticks(gender_arr, gender_MF, rotation = 0)
plt.legend(title='Target(Legend)', labels=labls, 
           loc='upper left', title_fontsize='11', fontsize='12')


# **Heart Disease Distribution based on Major Vessels Total**

# In[9]:


# Implementing Bar Chart
ax = pd.crosstab(HD_data.ca, HD_data.target).plot(figsize=(9, 5.3), kind='barh', color=[ '#FFD7D7', '#F17881'], 
                                         fontsize='10',
                                         edgecolor='#6D6A6A', alpha=0.85)
# Declaring Parameters for Bar Chart
for rect in ax.patches:
    wt, ht = rect.get_width(), rect.get_height()
    x, y = rect.get_xy()
    ax.text (x+wt/4, y+ht/4, wt, fontsize='8',
             horizontalalignment='left', verticalalignment='center')

plt.suptitle('Distribution of Heart Disease depending on Major Vessels Total', 
             fontweight='heavy', x=0.01, y=0.01, ha='left', fontsize='16', horizontalalignment='left',
             fontfamily='verdana', color='#100C07')
plt.title('Patients with 0 and 4 major vessels tend to have heart diseases. However, patients who have a number of vessels 1 to 3\ntend not to have heart diseases.', 
          fontsize='8', fontfamily='sans-serif', loc='left', color='#3E3B39')
plt.tight_layout(rect=[0.01, 0.10, 0.9, 1.1])
plt.xlabel('Total Number of People', color='#3E3B39', fontfamily='verdana', fontsize='8', fontweight='heavy')
plt.ylabel('Major Vessels in Number', color='#3E3B39', fontweight='heavy', 
            fontfamily='verdana')
plt.yticks(rotation=0)
labls = ['False', 'True']
plt.legend(title='Target(Legend)', labels=labls, loc='upper right', fontsize='11',
           title_fontsize='10')


# **Heart Disease Scatter Plot based on Age**

# In[10]:


# Declaring size for Scatter Plot
plt.figure(figsize=(5, 4))
plt.suptitle('Scatter Plot for Age-Based Heart Disease', color='#100C07', ha='left', fontweight='heavy', 
             x=0.04, y=0.01, fontfamily='verdana', 
             fontsize='14')
plt.title('Most of the patients who do and do not have heart disease are between 50 and 70. Compared to people without heart \n conditions, those with heart conditions tend to have higher heart rates.', 
          fontsize='8', fontfamily='verdana', loc='left', color='#3E3B39')
plt.tight_layout(rect=[0.01, 0.10, 0.9, 1.1])

# Implementing a scatter plot
plt.scatter(c='#8A0030', x=HD_data.age[HD_data.target==0], y=HD_data.thalach[(HD_data.target==0)])
plt.scatter(c='#FF5C8A', x=HD_data.age[HD_data.target==1], y=HD_data.thalach[(HD_data.target==1)])

# Declaring Legend and label for scatter plots
plt.legend(['False', 'True'], frameon=True,  title_fontsize='8.1', title='Type(legend)',
           loc='upper right', fontsize='8.2')
plt.xlabel('Age of a person', color='#3E3B39', fontweight='heavy',
           fontfamily='verdana', fontsize='13')
plt.ylabel('Maximum Heart Rate', color='#3E3B39', fontweight='heavy', 
           fontfamily='verdana', fontsize='13')
plt.show();


# # Heatmap

# In[11]:


# Correlation map or Heat Map
plt.figure(figsize=(7, 4.5))
sns.heatmap(HD_data.corr(), linewidths=0.13, annot=True, cmap='cool')
plt.suptitle('Correlation map or heat map for Numerical Variables', color='#100C07', fontweight='heavy', 
             x=0.03, y=0.03, ha='left', fontfamily='verdana', 
             fontsize='16')
plt.title('Resting blood pressure, cholestoral, and "oldpeak" have moderate relationship with age.', 
          fontsize='10', fontfamily='verdana', loc='left', color='#3E3B39')
plt.tight_layout(rect=[0.01, 0.10, 0.9, 1.1])


# # **Dataset Pre-processing**

# **One-Hot Encoding**

# In[12]:


# Getting dummy variables for the variables 'cp', 'slope' and 'thal'
cp = pd.get_dummies(HD_data['cp'], prefix='cp')
thal = pd.get_dummies(HD_data['thal'], prefix='thal')
slope = pd.get_dummies(HD_data['slope'], prefix='slope')

# adding the above dummy variables to original data set
add_dum = [HD_data, cp, thal, slope]
HD_data = pd.concat(add_dum, axis = 1)
HD_data.head()


# **Dropping Unnecessary Variables**

# In[13]:


# Drop variables 'cp', 'slope' and 'thal' as we are not using them
HD_data = HD_data.drop(columns = ['cp', 'thal', 'slope'])
HD_data.head()


# **Spiliting and Normalizing the data**

# In[14]:


# dropping the target column
x = HD_data.drop(['target'], axis=1)
y = HD_data['target']
# Normalising the data using min max scaler
x = MinMaxScaler().fit_transform(x)
# splitting dataset in to 80% as train and 20% as test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


# # **Model Implementation**

# In[15]:


# Applying Logistic Regression, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier to the data
Models = ['LogisticRegression', 'KNeighborsClassifier','RandomForestClassifier','GradientBoostingClassifier','']
classifiers = [LogisticRegression(solver='liblinear', max_iter=999, random_state=1, penalty='l1'), KNeighborsClassifier(n_neighbors=3),
               RandomForestClassifier(n_estimators=999, max_leaf_nodes=20, random_state=1, min_samples_split=15),
               GradientBoostingClassifier(max_leaf_nodes=3, random_state=1, n_estimators=102, min_samples_leaf=20, loss='exponential')]
print('Model Accuracies: ')
Accuracy = []
for i in range(len(classifiers)):
  classifier = classifiers[i] 
  classifier.fit(x_train, y_train)
  y_pred = classifier.predict(x_test)
  confmat = confusion_matrix(y_pred, y_test)
  Acc = accuracy_score(y_pred, y_test)
  Accuracy.append(Acc)
  
  print('\n' + Models[i] + '->', Acc*100)


# **Model Comparison**

# In[16]:


# Comparing all the implemented models
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Random Forest', 'Gradient Boosting'], 
                        'Accuracy': [Accuracy[0]*100, Accuracy[1]*100, Accuracy[2]*100, Accuracy[3]*100]})

print("Highest accuracy we observed in: "+ "\033[1m"+Models[pd.Series(Accuracy).idxmax()]+"\033[0m" )
compare.sort_values(by='Accuracy', ascending=False).style.background_gradient(cmap='BuGn').hide_index().set_properties(**{'font-family': 'verdana'})


# # **Hybrid Machine Learning Model**

# In[17]:


Hybrid = VotingClassifier(estimators=[('knn', KNeighborsClassifier(n_neighbors=2)), ('rf', RandomForestClassifier(n_estimators=2000, max_leaf_nodes=20, random_state=1, min_samples_split=20))
, ('gnb', GradientBoostingClassifier(random_state=1, max_leaf_nodes=3, n_estimators=2000, loss='exponential', min_samples_leaf=20))], voting='soft')
Hybrid = Hybrid.fit(x_train, y_train)
y_pred_hybrid = Hybrid.predict(x_test)
Acc_hybrid = accuracy_score(y_pred_hybrid, y_test)

print('Hybrid_Model accuracy is', Acc_hybrid*100)

