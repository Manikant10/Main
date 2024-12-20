#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("LOAN_UPDATED.csv")


# In[2]:


df.shape


# In[3]:


df.columns


# In[4]:


X=df.drop(['loan_id',' loan_status'],axis=1)


# In[5]:


y=df[' loan_status']


# In[6]:




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
mode=DecisionTreeClassifier()
mode.fit(X_train,y_train)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(X_train,y_train)


# In[13]:


m.score(X_test,y_test)


# In[14]:


mode.score(X_test,y_test)


# In[15]:


kn.score(X_test,y_test)


# In[ ]:





# In[16]:


from sklearn.ensemble import RandomForestClassifier
import joblib

# Train a model
model = RandomForestClassifier()
model.fit(X_train,y_train)

# Save the model to a file
joblib.dump(model, 'model.pkl')

# Load the model from the file


# In[17]:


import joblib

try:
    model = joblib.load('model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")


# In[18]:



# In[ ]:


from flask_cors import CORS
from flask import Flask,request,jsonify
import pickle 

app=Flask(__name__)
CORS(app)

from flask_cors import CORS

model=pickle.load(open('model.pkl','rb'))
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json()
        input_data=data.get('inputData')
        prediction=model.prediction([input_data])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ =='__main__':
    app.run(host='0.0.0.0', port=200)


# In[ ]:


import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model
# Try to load the model
try:
    model = joblib.load('iris_model.pkl')
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





# In[ ]:




