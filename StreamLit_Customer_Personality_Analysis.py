import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import streamlit as st
from annotated_text import annotated_text 

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os

st.set_page_config(page_title="Streamlit_Customer_Personality_Analysis", page_icon="🎉",layout="wide")

np.random.seed(42)

#Side bar

with st.sidebar:
    st.title("Streamlit Customer Personality Analysis")



st.write('Welcome to My Streamlit Web Application')


df = pd.read_csv("./data/marketing_campaign.csv",sep="\t")



if st.checkbox("Show Codes and comments"):
  
    # display the code in page
    st.markdown('---')
    st.subheader('Code')
    body ="""df = df.dropna()
    df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64[ns]')
    Membership_Period = []
    for i in df["Dt_Customer"]:
        j = df["Dt_Customer"].max() - i
        Membership_Period.append(j)
    df['Membership_Period'] = Membership_Period

    df['Membership_Period'] = pd.to_numeric(df['Membership_Period'].dt.days, downcast='integer')

    df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'Couple')
    df['Marital_Status'] = df['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'],'Single')

    df['Education'] = df['Education'].replace(['PhD','Master'],'Postgraduate')
    df['Education'] = df['Education'].replace(['Graduation'],'Graduate')
    df['Education'] = df['Education'].replace(['2n Cycle','Basic'],'UnderGraduate')

    column_names = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts','MntGoldProds']
    df['Total_Spent']= df[column_names].sum(axis=1)


    column_names_2 = ['Kidhome', 'Teenhome']
    df['Total_children'] = df[column_names_2].sum(axis=1)


    column_names_3 = ['AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2']
    df['Total_Cmp'] = df[column_names_3].sum(axis=1)


    column_names_4 = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[column_names_4].sum(axis=1)

    family_size = []
    for i in df['Marital_Status']:
        if i == 'Single':
            family_size.append(1)
        else:
            family_size.append(2)

    df['Family_Size'] = family_size + df['Total_children']

    df["Age"] = 2022-df["Year_Birth"]
    drop = [ "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    df = df.drop(drop, axis=1)
    df["Family"] = np.where(df.Total_children> 0, 1, 0)


    df = df[(df["Age"]<90)]
    df = df[(df["Income"]<600000)]


    df_2 = df.drop(columns=['Kidhome', 'Teenhome','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain','NumDealsPurchases', 'NumWebPurchases',
        'NumCatalogPurchases', 'NumStorePurchases'])

    df_2 = df_2.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts','MntGoldProds'])
    """
    st.code(body,language='python')






df = df.dropna()
df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64[ns]')
Membership_Period = []
for i in df["Dt_Customer"]:
    j = df["Dt_Customer"].max() - i
    Membership_Period.append(j)
df['Membership_Period'] = Membership_Period

df['Membership_Period'] = pd.to_numeric(df['Membership_Period'].dt.days, downcast='integer')

df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'],'Single')

df['Education'] = df['Education'].replace(['PhD','Master'],'Postgraduate')
df['Education'] = df['Education'].replace(['Graduation'],'Graduate')
df['Education'] = df['Education'].replace(['2n Cycle','Basic'],'UnderGraduate')

column_names = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts','MntGoldProds']
df['Total_Spent']= df[column_names].sum(axis=1)


column_names_2 = ['Kidhome', 'Teenhome']
df['Total_children'] = df[column_names_2].sum(axis=1)


column_names_3 = ['AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2']
df['Total_Cmp'] = df[column_names_3].sum(axis=1)


column_names_4 = ['NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[column_names_4].sum(axis=1)


family_size = []
for i in df['Marital_Status']:
    if i == 'Single':
        family_size.append(1)
    else:
        family_size.append(2)

df['Family_Size'] = family_size + df['Total_children']
df["Age"] = 2022-df["Year_Birth"]
drop = [ "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(drop, axis=1)
df["Family"] = np.where(df.Total_children> 0, 1, 0)
df = df[(df["Age"]<90)]
df = df[(df["Income"]<600000)]
df_2 = df.drop(columns=['Kidhome', 'Teenhome','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain','NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases'])
df_2 = df_2.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts','MntSweetProducts','MntGoldProds'])



if st.checkbox("Report"):
    st.info("""
    This is a report created with pandas profiling, it will show you the dataframe's structure,
    """)

    pr = ProfileReport(df_2,minimal = True)

    st_profile_report(pr)

    

