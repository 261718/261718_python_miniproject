#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('Downloads/diabetes.csv')
df.plot(kind = 'Box' , figsize = (20,10))
plt.show()

# In[18]:


df = df[df['SkinThickness'] < 80]
df = df[df['Insulin'] <= 600]
print(df.shape)

# In[19]:


df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean()
df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean()
df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean()
df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean()
df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean()

print(df.head(10))
df.to_csv("Downloads/CleanedData.csv")


# In[20]:


import seaborn as sns
plt.figure(1)
plt.subplot(121), sns.distplot(df['Glucose'])
plt.subplot(122), df['Glucose'].plot.box(figsize=(16,5))
plt.show()


# In[21]:


plt.figure(2)
plt.subplot(121), sns.distplot(df['Pregnancies'])
plt.subplot(122), df['Pregnancies'].plot.box(figsize=(16,5))
plt.show()


# In[22]:


plt.subplot(121), sns.distplot(df['BloodPressure'])
plt.subplot(122), df['BloodPressure'].plot.box(figsize=(16,5))
plt.show()


# In[23]:


plt.subplot(121), sns.distplot(df['Insulin'])
plt.subplot(122), df['Insulin'].plot.box(figsize=(16,5))
plt.show()


# In[24]:


plt.subplot(121), sns.distplot(df['SkinThickness'])
plt.subplot(122), df['SkinThickness'].plot.box(figsize=(16,5))
plt.show()


# In[25]:


plt.subplot(121), sns.distplot(df['DiabetesPedigreeFunction'])
plt.subplot(122), df['DiabetesPedigreeFunction'].plot.box(figsize=(16,5))
plt.show()


# In[26]:


plt.subplot(121), sns.distplot(df['BMI'])
plt.subplot(122), df['BMI'].plot.box(figsize=(16,5))
plt.show()


# In[27]:


plt.subplot(121), sns.distplot(df['Age'])
plt.subplot(122), df['Age'].plot.box(figsize=(16,5))
plt.show()


# In[28]:


df=df.drop(["Insulin", "SkinThickness", "Age"], axis=1)
df = df/df.max()
print(df.head(5))
df.to_csv("Downloads/Normalized.csv")


# In[29]:


x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(x.head())


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics


y = df['Outcome']
x = df.drop(columns=['Outcome'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=39)

svcmodel = SVC(probability=True)

                                            
svcmodel.fit(x_train, y_train)

prediction = svcmodel.predict(x_test)

acc = metrics.accuracy_score(y_test, prediction)
print(acc)


# In[32]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, prediction)


# In[33]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, prediction)


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))


# In[35]:


# The confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

svmcla_cm = confusion_matrix(y_test, prediction)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(svmcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('SVM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[36]:


from sklearn.tree import DecisionTreeClassifier

# define the model
dtcla = DecisionTreeClassifier(random_state=9)

# train model
dtcla.fit(x_train, y_train)

# predict target values
y_predict4 = dtcla.predict(x_test)


# In[37]:


# Test score
score_dtcla = dtcla.score(x_test, y_test)
print(score_dtcla)


# In[38]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict4)


# In[39]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict4)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict4))


# In[41]:


# The confusion matrix
dtcla_cm = confusion_matrix(y_test, y_predict4)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(dtcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Decision Tree Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[42]:


from sklearn.naive_bayes import GaussianNB

# We define the model
nbcla = GaussianNB()

# We train model
nbcla.fit(x_train, y_train)

# We predict target values
y_predict3 = nbcla.predict(x_test)


# In[43]:


# Test score
score_nbcla = nbcla.score(x_test, y_test)
print(score_nbcla)


# In[44]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_predict3)


# In[45]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict3)


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict3))


# In[47]:


# The confusion matrix
nbcla_cm = confusion_matrix(y_test, y_predict3)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(nbcla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Naive Bayes Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# In[48]:


import matplotlib.pyplot as plt
x=['SVM','Decision Tree','NaviBayes']
y=[90.9,77.9,87]
plt.bar(x,y)
plt.title('Bar Chart')
plt.xlabel('ML Algorithms')
plt.ylabel('Accuracies')
plt.show()


# In[49]:


def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[50]:


from sklearn.metrics import roc_curve
# SVM Classification
Y_predict2_proba = svcmodel.predict_proba(x_test)
Y_predict2_proba = Y_predict2_proba[:, 1]
fper, tper, thresholds = roc_curve(y_test, Y_predict2_proba)
plot_roc_cur(fper, tper)


# In[51]:


from sklearn.metrics import roc_curve
# Decision Tree Classification
Y_predict2_proba = dtcla.predict_proba(x_test)
Y_predict2_proba = Y_predict2_proba[:, 1]
fper, tper, thresholds = roc_curve(y_test, Y_predict2_proba)
plot_roc_cur(fper, tper)


# In[52]:


from sklearn.metrics import roc_curve
# NaviBayes Classification
Y_predict2_proba = nbcla.predict_proba(x_test)
Y_predict2_proba = Y_predict2_proba[:, 1]
fper, tper, thresholds = roc_curve(y_test, Y_predict2_proba)
plot_roc_cur(fper, tper)


# In[ ]:





# In[ ]:





# In[ ]:




