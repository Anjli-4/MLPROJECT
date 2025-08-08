# .house py file me
import streamlit as st
import pandas as pd
import random
import pickle
import warnings



from sklearn.preprocessing import StandardScaler
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price prediction')
# center image
st.image('https://user-images.githubusercontent.com/72307306/186478948-2703db82-9c02-4b45-8a6e-c09801418526.png')       

st.header('Model of housing prices to predict median house values in California')
# st.subheader('''User must Enter given values to Predict Price:
#             ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title('Select House Features')
# left size image
st.sidebar.image('https://www.fingerprinthomes.com/wp-content/uploads/2021/08/dream-elevation-A.jpg')
temp_df=pd.read_csv('California.csv')

random.seed(22)
all_values=[]

for i in temp_df[col]:
  min_value,max_value = temp_df[i].agg(['min','max'])

  var = st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
                   random.randint(int(min_value),int(max_value)))
  all_values.append(var)

ss=StandardScaler()
ss.fit(temp_df[col])


final_value = ss.transform([all_values])

# here imported the model
# import pickle
with open('house_price_pred_ridge_model.pkl','rb') as f:
  chatgpt= pickle.load(f)

price= chatgpt.predict(final_value)[0] #it will call the 0th index value of array

import time

#here  zip adds both col and all_values
st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))   
value=0
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')


# to make gif img
place= st.empty()
place.image('https://cdn.dribbble.com/users/1501052/screenshots/5468049/searching_tickets.gif',width=150)
if price>0:
    # progress_bar=st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress_bar.progress(i+1)


    body=f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty() 
    st.success(body)

else:
    body='Invalid House Features Values'
    st.warning(body)
    
