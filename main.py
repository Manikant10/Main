#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("C:\\loan.csv")


# In[2]:


df.shape


# In[3]:


df.columns


# In[4]:


X=df.drop(['loan_id',' loan_status'],axis=1)


# In[5]:


y=df[' loan_status']


# In[6]:


X=pd.get_dummies(X)


# In[7]:


X.shape


# In[8]:


X.columns


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=35)


# In[10]:


from sklearn.linear_model import LogisticRegression
m=LogisticRegression()
m.fit(X_train,y_train)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
prediction=DecisionTreeClassifier()
prediction.fit(X_train,y_train)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(X_train,y_train)


# In[13]:


m.score(X_test,y_test)


# In[16]:


prediction.score(X_test,y_test)


# In[18]:


kn.score(X_test,y_test)


# In[22]:


import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris



# Save the trained model to a file
joblib.dump(prediction, 'loan_model.pkl')


# In[24]:


import joblib

try:
    model = joblib.load('loan_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")


# In[ ]:


import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

# Try to load the model
try:
    model = joblib.load('loan_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set to None or handle accordingly

@app.route('/predict', methods=['POST'])
def a_p():
    if model is None:
        return jsonify({'error': 'Model failed to load'}), 500

    try:
        # Assume the input is JSON with 'features' key
        data = request.get_json()
        features = X

        prediction = model.predict(X)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Handle input or prediction error

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except SystemExit as e:
        print(f"SystemExit: {e}")


# In[ ]:



# In[ ]:




