#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from tkinter import ttk


# In[2]:


df= pd.read_csv('/Users/nirmalsingh/Desktop/Bank Projects/creditcard.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# ## Clearly from Above output it can be observed that All columns have values in almost same range except that Amount Column have values with a larger scale ,So We have to perform Column Standardization at Amount Column.

# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


s=StandardScaler() # Instance of StandardScaler is made


# In[7]:


df['Amount']= s.fit_transform(pd.DataFrame(df['Amount']))


# In[8]:


df.head()


# ## Missing Values Check 

# In[9]:


df.isna().sum()


# ## Duplicate Check and Removal

# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.duplicated().sum()


# In[13]:


df.shape


# In[14]:


df.head()


# ## For my Analytics I dont need Time Column in my Data Frame 

# In[15]:


df.drop(['Time'],axis=1,inplace=True)


# In[16]:


df.head()


# In[17]:


df.shape


# In[18]:


df['Class'].value_counts()


# In[19]:


import seaborn as sns 
sns.countplot(df['Class'])


# ## Clearly the dataset is heavily imbalanced 

# There are two techniques we can use to handle imbalanced data set.
# 
# 1.UnderSampling 
# 
# 2.OverSampling 

# In[20]:


normal_transactions = df[df['Class']==0]
fraud_transactions =df[df['Class']==1]


# In[21]:


normal_transactions.shape


# In[22]:


fraud_transactions.shape


# In[23]:


normal_transactions_sample=normal_transactions.sample(473)


# In[24]:


normal_transactions_sample.shape


# In[25]:


new_data=pd.concat([normal_transactions_sample,fraud_transactions])


# In[26]:


new_data['Class'].value_counts()


# In[ ]:





# ## Storing Feature Matrix in Vector X and Class (Response) in Vector Y

# In[27]:


x=df.drop('Class',axis=1)
y=df['Class']


# In[28]:


x


# In[29]:


y


# ## Splitting the Data Set as Training and Testing Data Set 

# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=40)


# ## Logistic Regression fitting to the training DataSet

# In[31]:


from sklearn.linear_model import LogisticRegression

# Create an instance of the LogisticRegression model
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(x_train, y_train)



# In[32]:


# Predict using the trained model
y_pred = logreg.predict(x_train)


# In[33]:


# Calculate accuracy on the testing dataset
accuracy = logreg.score(x_test, y_test)

# Print the accuracy
print("Accuracy on testing dataset:", accuracy)


# In[34]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Predict using the trained model on the testing dataset
y_pred = logreg.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, y_pred)

# Calculate recall score
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Print the scores
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# After UnderSampling 
# 

# In[35]:


x1=new_data.drop('Class',axis=1)
y1=new_data['Class']


# In[36]:


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20, random_state=42)


# In[37]:


from sklearn.linear_model import LogisticRegression

# Create an instance of the LogisticRegression model
logreg1 = LogisticRegression()

# Fit the model to the training data
logreg1.fit(x1_train, y1_train)


# In[38]:


# Predict using the trained model
y1_pred = logreg1.predict(x1_train)


# In[39]:


# Calculate accuracy on the testing dataset
accuracy = logreg1.score(x1_test, y1_test)

# Print the accuracy
print("Accuracy on testing dataset:", accuracy)


# In[40]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Predict using the trained model on the testing dataset
y1_pred = logreg1.predict(x1_test)

# Calculate precision score
precision1 = precision_score(y1_test, y1_pred)

# Calculate recall score
recall1 = recall_score(y1_test, y1_pred)

# Calculate F1 score
f1_1 = f1_score(y1_test, y1_pred)

# Print the scores
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_1)


# Clearly all the three scores have imporved after imbalancing of the data set is dealt 

# In[41]:


a= LogisticRegression()
a.fit(x1,y1)


# ## Save the Model

# In[42]:


import joblib


# In[43]:


joblib.dump(a,"Fraud Detection Model")


# In[44]:


model = joblib.load("Fraud Detection Model")


# In[45]:


pred=model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
pred


# In[46]:


if pred==0:
    print("Normal Transaction")
else:
    print("Fraud Transaction")
    


# In[47]:


import tkinter as tk
from tkinter import messagebox


# In[48]:


# Create a GUI for input and prediction
def predict_transaction():
    try:
        # Retrieve input values from the GUI
        v1 = float(entry_v1.get())
        v2 = float(entry_v2.get())
        v3 = float(entry_v3.get())
        v4 = float(entry_v4.get())
        v5 = float(entry_v5.get())
        v6 = float(entry_v6.get())
        v7 = float(entry_v7.get())
        v8 = float(entry_v8.get())
        v9 = float(entry_v9.get())
        v10 = float(entry_v10.get())
        v11 = float(entry_v11.get())
        v12 = float(entry_v12.get())
        v13 = float(entry_v13.get())
        v14 = float(entry_v14.get())
        v15 = float(entry_v15.get())
        v16 = float(entry_v16.get())
        v17 = float(entry_v17.get())
        v18 = float(entry_v18.get())
        v19 = float(entry_v19.get())
        v20 = float(entry_v20.get())
        v21 = float(entry_v21.get())
        v22 = float(entry_v22.get())
        v23 = float(entry_v23.get())
        v24 = float(entry_v24.get())
        v25 = float(entry_v25.get())
        v26 = float(entry_v26.get())
        v27 = float(entry_v27.get())
        v28 = float(entry_v28.get())
        v29 = float(entry_v29.get())
        # ... Retrieve other input values as needed

        # Load the trained model
        model = joblib.load("Fraud Detection Model")

        # Perform prediction
        pred = model.predict([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29]])

        if pred == 0:
            messagebox.showinfo("Prediction Result", "Normal Transaction")
        else:
            messagebox.showinfo("Prediction Result", "Fraud Transaction")
    except ValueError:
        messagebox.showerror("Error", "Invalid input values.")

# Create the GUI window
window = tk.Tk()
window.title("Transaction Fraud Detection")
window.geometry("150x150")

# Create a canvas and scrollbars
canvas = tk.Canvas(window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_y1 = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y1.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar_y1.set)

# Create a frame to contain the GUI elements
frame = tk.Frame(canvas)

# Add padding to the frame to ensure proper scrolling
frame.pack(pady=10)

# Add the frame as the canvas's window
canvas.create_window((0, 0), window=frame, anchor="nw")

# Create entry fields for input
label_v1 = tk.Label(frame, text="Input v1:")
label_v1.pack()
entry_v1 = tk.Entry(frame)
entry_v1.pack()

label_v2 = tk.Label(frame, text="Input v2:")
label_v2.pack()
entry_v2 = tk.Entry(frame)
entry_v2.pack()

label_v3 = tk.Label(frame, text="Input v3:")
label_v3.pack()
entry_v3 = tk.Entry(frame)
entry_v3.pack()

label_v4 = tk.Label(frame, text="Input v4:")
label_v4.pack()
entry_v4 = tk.Entry(frame)
entry_v4.pack()

label_v5 = tk.Label(frame, text="Input v5:")
label_v5.pack()
entry_v5 = tk.Entry(frame)
entry_v5.pack()

label_v6 = tk.Label(frame, text="Input v6:")
label_v6.pack()
entry_v6 = tk.Entry(frame)
entry_v6.pack()

label_v7 = tk.Label(frame, text="Input v7:")
label_v7.pack()
entry_v7 = tk.Entry(frame)
entry_v7.pack()

label_v8 = tk.Label(frame, text="Input v2:")
label_v8.pack()
entry_v8 = tk.Entry(frame)
entry_v8.pack()

label_v9 = tk.Label(frame, text="Input v9:")
label_v9.pack()
entry_v9 = tk.Entry(frame)
entry_v9.pack()

label_v10 = tk.Label(frame, text="Input v10:")
label_v10.pack()
entry_v10 = tk.Entry(frame)
entry_v10.pack()

label_v11 = tk.Label(frame, text="Input v11:")
label_v11.pack()
entry_v11 = tk.Entry(frame)
entry_v11.pack()

label_v12 = tk.Label(frame, text="Input v12:")
label_v12.pack()
entry_v12 = tk.Entry(frame)
entry_v12.pack()

label_v13 = tk.Label(frame, text="Input v13:")
label_v13.pack()
entry_v13 = tk.Entry(frame)
entry_v13.pack()

label_v14 = tk.Label(frame, text="Input v14:")
label_v14.pack()
entry_v14 = tk.Entry(frame)
entry_v14.pack()

label_v15 = tk.Label(frame, text="Input v15:")
label_v15.pack()
entry_v15 = tk.Entry(frame)
entry_v15.pack()

label_v16 = tk.Label(frame, text="Input v16:")
label_v16.pack()
entry_v16 = tk.Entry(frame)
entry_v16.pack()

label_v17 = tk.Label(frame, text="Input v17:")
label_v17.pack()
entry_v17 = tk.Entry(frame)
entry_v17.pack()

label_v18 = tk.Label(frame, text="Input v18:")
label_v18.pack()
entry_v18 = tk.Entry(frame)
entry_v18.pack()

label_v19 = tk.Label(frame, text="Input v19:")
label_v19.pack()
entry_v19 = tk.Entry(frame)
entry_v19.pack()

label_v20 = tk.Label(frame, text="Input v20:")
label_v20.pack()
entry_v20 = tk.Entry(frame)
entry_v20.pack()

label_v21 = tk.Label(frame, text="Input v21:")
label_v21.pack()
entry_v21 = tk.Entry(frame)
entry_v21.pack()


label_v22 = tk.Label(frame, text="Input v22:")
label_v22.pack()
entry_v22 = tk.Entry(frame)
entry_v22.pack()

label_v23 = tk.Label(frame, text="Input v23:")
label_v23.pack()
entry_v23 = tk.Entry(frame)
entry_v23.pack()

label_v24 = tk.Label(frame, text="Input v24:")
label_v24.pack()
entry_v24 = tk.Entry(frame)
entry_v24.pack()

label_v25 = tk.Label(frame, text="Input v25:")
label_v25.pack()
entry_v25 = tk.Entry(frame)
entry_v25.pack()

label_v26 = tk.Label(frame, text="Input v26:")
label_v26.pack()
entry_v26 = tk.Entry(frame)
entry_v26.pack()

label_v27 = tk.Label(frame, text="Input v27:")
label_v27.pack()
entry_v27 = tk.Entry(frame)
entry_v27.pack()

label_v28 = tk.Label(frame, text="Input v28:")
label_v28.pack()
entry_v28 = tk.Entry(frame)
entry_v28.pack()

label_v29 = tk.Label(frame, text="Input v29:")
label_v29.pack()
entry_v29 = tk.Entry(frame)
entry_v29.pack()

# Create a button for prediction
predict_button = tk.Button(frame, text="Predict", command=predict_transaction)
predict_button.pack()

# Update the canvas scrolling region
frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Bind the scrolling action to the canvas
canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

# Run the GUI event loop
window.mainloop()


# In[ ]:




