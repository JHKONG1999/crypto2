#----!AUTHORS NAME: KONG JUN HAO----
#----IMPORTS LIBRARIES----
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plost

from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import plotly.express as px
import io 
from sklearn.ensemble import RandomForestClassifier

#---IMPORT DATASET & DEFINE TARGET---
crypto = pd.read_csv('attack0.95.csv')

df = crypto.copy()
target = 'attack'
encode = ['fix_size','mem_available','mem_cached','mem_free','mem_inactive','memswap_free','memswap_total']

#---SEPERATING X AND Y---
X = df.drop('attack', axis=1)
Y = df['attack']

#----PAGE SETTINGS-----
st.set_page_config(page_title="CRYPTOJACKING DETECTION APP",layout="wide",initial_sidebar_state="expanded")

#----OPEN CSS FILE----
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    
#----NAVIGATION BAR-----
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #000000;">
  <a class="navbar-brand" href="#">CRYPTOJACKING</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#"><span class="sr-only">(current)</span></a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#-----HORIZONTAL MENU----
selected =option_menu(
  menu_title=None,
  options=["Dashboard","Predicts","Heatmap"],
  icons=["house","shield-check","graph-up"],
  menu_icon="cast",
  default_index=0,
  orientation="horizontal",
  styles={
          "container": {"padding": "0!important"},
          "icon": {"font-size": "16px"},
          "nav-link": {
              "font-size": "16px",
              "text-align": "center",
              "margin": "0px",
              "--hover-color": "#f4cccc",
          },
          "nav-link-selected": {"background-color": "#e06666"},
      },
)

#---HEADER----
st.write("""
# **CRYPTOJACKING DETECTION SYSTEM**
**INTRODUCTION:**
* This app predicts the **CRYPTOJACKING** species!
""")

#----SIDEBAR-----
st.sidebar.image(Image.open('dl-ump-logo.png'))
st.sidebar.header('**CPU Memory**')
st.sidebar.subheader('User Input Features')
def user_input_features():
        fix_size = st.sidebar.text_input('fix_size', '0')
        mem_available = st.sidebar.text_input('mem_available', '0')
        mem_cached = st.sidebar.text_input('mem_cached', '0')
        mem_free = st.sidebar.text_input('mem_free', '0')
        mem_inactive = st.sidebar.text_input('mem_inactive', '0')
        memswap_free = st.sidebar.text_input('memswap_free', '0')
        memswap_total = st.sidebar.text_input('memswap_total', '0')
        
        data = {'fix_size': fix_size,
                'mem_available': mem_available,
                'mem_cached': mem_cached,
                'mem_free': mem_free,
                'mem_inactive': mem_inactive,
                'memswap_free': memswap_free,
                'memswap_total': memswap_total
                }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

#---FOOTER OF THE SIDEBAR---
st.sidebar.markdown('''
---
Created by **Kong Jun Hao** associated with [Universiti Malaysia Pahang (UMP)](https://www.ump.edu.my/en).
''')


#---DAHSBOARD INRTERFACE----
if selected =="Dashboard":
  #Display dashboard
  st.markdown('### **Metrics**')
  col1, col2, col3 = st.columns(3)
  col1.metric("fix size", "4708106240","4%")
  col2.metric("memory available", "1092403200","10%")
  col3.metric("memory cached", "415768576","8%")
  
  col4, col5, col6 = st.columns(3)
  col4.metric("memory free", "1092403200","12%")
  col5.metric("memory inactive", "267345920","5%")
  col6.metric("memory swap free", "645918720","15%")

  col7,col8, col9 = st.columns(3)
  col7.metric("memory swap total", "645918720","12%")
  col8.metric("--","---")
  col9.metric("--","---")

  #Line chart
  st.write("##")
  st.markdown('### **Line Chart**')
  crypto = pd.DataFrame(
      np.random.randn(100,7),
      columns=['fix_size', 'mem_available', 'mem_cached','mem_free','mem_inactive','memswap_free','memswap_total'])

  st.line_chart(crypto)


#----PREDICTS INTERFACE----
if selected =="Predicts":
  # Displays the user input features
  st.subheader('**User Input Features**')
  st.write(input_df)
  crypto = pd.read_csv('attack0.95.csv')

  # Build random forest model
  clf = RandomForestClassifier()
  clf.fit(X, Y)

  # Saving the model
  import pickle
  pickle.dump(clf, open('crypto_clf.pkl', 'wb'))

  # Reads in saved classification model
  load_clf = pickle.load(open('crypto_clf.pkl', 'rb'))

  a1,a2=st.columns([2,8])
  with a1:
  #Prediction Result
    prediction = clf.predict(input_df)
    prediction_proba = clf.predict_proba(input_df)

    st.write("##")
    st.subheader('**Result**')
    st.write(prediction)

  with a2:
    st.write("##")
    st.subheader('**Prediction Probability**')
    st.write(prediction_proba)

  st.write("##")
  st.write("**The result :**")
  st.write("1 indicates **THE DEVICES IS SAFE** !")
  st.write("0 indicates **THE DEVICES IS HIJACKED** !")
  st.write("##")


#---HEATMAP INTERFACE---
if selected =="Heatmap":
  c1,c2=st.columns((8,3))
  with c1:
  #Heatmap
    st.subheader("**Intercorrelation Matrix Heatmap**")
    crypto = pd.read_csv('attack0.95.csv')

    corr = crypto.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("darkgrid"):
      fig, ax = plt.subplots(figsize=(15,8))
      ax = sns.heatmap(corr,mask=mask, vmax=1, square=True)

    st.pyplot(fig)

#---HIDE STREAMLIT STYLE----
hide_style = """
  <style>
  MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

#---BOOTSTRAP----
st.markdown("""
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
""", unsafe_allow_html=True)
