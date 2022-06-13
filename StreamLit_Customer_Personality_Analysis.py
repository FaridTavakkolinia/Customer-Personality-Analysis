from re import A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


np.random.seed(42)

import streamlit as st
from annotated_text import annotated_text 

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys
import os

st.set_page_config(page_title="Streamlit_Customer_Personality_Analysis", page_icon="🎉",layout="centered")

np.random.seed(42)

#Side bar

annotated_text(("Streamlit Customer Personality Analysis","🎉", ))


with st.sidebar:
    st.title("Streamlit Customer Personality Analysis")
    
    st.markdown("This app is a Streamlit app that uses the Streamlit framework to analyze the customer personality of a company.")
    st.subheader("About:")
    st.subheader("Farid Tavakkolinia")
    st.subheader("Farid.Tavakkolinia@studenti.univr.it")
    st.subheader("GitHub [link](https://github.com/FaridTavakkolinia/Customer-Personality-Analysis)")

    


st.image('douglas.jpg',width=600,use_column_width=True)

st.header('Welcome to My Streamlit Web Application')
st.markdown("""----""")
st.write('in This App we will be analysing the Customer Personality and will be able to predict the Customer behaviour and Create a clustring model')


df = pd.read_csv("./data/marketing_campaign.csv",sep="\t")

st.subheader("In the Below you can See the Dataset of the Customer Personalyty Analysis Head")
st.dataframe(df.head())
st.markdown("""----""")
st.subheader("Describing the Dataset")
st.dataframe(df.describe())
st.markdown("""----""")
st.subheader('Display missing values of DataSet as a picture using missingno ')

st.image('MissingValues.jpg',caption='Missing Values As a Picture',width=900)
st.write("As you can see we have some missing values in Income Column ")


#working on the data 


df = df.dropna()


df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64[ns]')


Membership_Period = []
for i in df["Dt_Customer"]:
    j = df["Dt_Customer"].max() - i
    Membership_Period.append(j)

df['Membership_Period'] = Membership_Period
df['Membership_Period'] = pd.to_numeric(df['Membership_Period'].dt.days, downcast='integer')



df["Marital_Status"].value_counts()

df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'],'Single')
df["Marital_Status"].value_counts()

df["Education"].value_counts()
df['Education'] = df['Education'].replace(['PhD','Master'],'Postgraduate')
df['Education'] = df['Education'].replace(['Graduation'],'Graduate')
df['Education'] = df['Education'].replace(['2n Cycle','Basic'],'UnderGraduate')
df["Education"].value_counts()


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


To_Plot = [ "Income", "Recency", "Education", "Age", "Total_Spent", "Family_Size","Membership_Period","Family"]


df = df[(df["Age"]<90)]
df = df[(df["Income"]<600000)]


c = df.select_dtypes(include='object').columns


labelencoder = LabelEncoder()
df['Education'] = labelencoder.fit_transform(df['Education'])
df['Marital_Status'] = labelencoder.fit_transform(df['Marital_Status'])



df_2 = df.drop(columns=['Kidhome', 'Teenhome','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain','NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases'])



scaler = StandardScaler()
scaler.fit(df_2)
df_2_scaled = pd.DataFrame(scaler.transform(df_2),columns= df_2.columns )


corrmat= df_2_scaled.corr()
plt.figure(figsize=(20,15))  
sns.heatmap(corrmat,annot=True)

pca = PCA(n_components=3)
pca.fit(df_2_scaled)
PCA_df_2_scaled = pd.DataFrame(pca.transform(df_2_scaled), columns=(["column_1","column_2", "column_3"]))



kmeancluster = KMeans(n_clusters=4,random_state=42)
PCA_df_2_scaled["Cluster"] = kmeancluster.fit_predict(PCA_df_2_scaled)



labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

cluster1_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==0].shape[0]
cluster2_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==1].shape[0]
cluster3_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==2].shape[0]
cluster4_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==3].shape[0]
values = [cluster1_val, cluster2_val, cluster3_val, cluster4_val]






#Plotly Histogram

st.subheader("""Histogram of Age which seprated bye marital status""")
fig = px.histogram(df, x="Age",color="Marital_Status", labels={"Age": "Age", "Marital_Status": "Marital Status"},height=500, width=1000)
st.plotly_chart(fig)


st.markdown("""----""")

st.subheader("""Sccater plot of Income and Total Spent which size of dots increase by Family Size and seperated by Education""")

fig = px.scatter(df, x="Income", y="Total_Spent",size="Family_Size",  color="Education" , height=500, width=1000,)
st.plotly_chart(fig)




with st.container():
    st.markdown("""----""")
    st.subheader("""Correlation matrix of all the features after scalling""")
    corrmat= df_2_scaled.corr()
    fig, ax = plt.subplots( figsize=(20,15))
    sns.heatmap(corrmat,ax=ax,annot=True, cmap="RdBu_r",fmt='.2f')
    st.pyplot(fig)


st.markdown("""----""")
st.subheader("""3D Scatter Plot of all """)
fig = px.scatter_3d(PCA_df_2_scaled, x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"], z=PCA_df_2_scaled["column_3"],)
st.plotly_chart(fig)


with st.container():

    st.markdown("""----""")
    st.write('In the blow plot we can see the distribution of clusters')
    fig, ax = plt.subplots( figsize=(8,4))
    sns.countplot(data=PCA_df_2_scaled, x="Cluster", palette="tab20")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    st.pyplot(fig)



st.markdown("""----""")
#Plot of Clusters
with st.container():
    st.write("Plot of Clusters")
    fig, ax = plt.subplots( figsize=(8,4))
    pl = sns.scatterplot(data = PCA_df_2_scaled,x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"],hue=PCA_df_2_scaled["Cluster"], palette="tab20")
    pl.set_title("Cluster")
    plt.legend()
    st.pyplot(fig)



st.markdown("""----""")
with st.container():
    st.write("Pie Chart of Clusters")
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole = 0.5, title="Clusters")]) # Create a pie chart
    st.plotly_chart(fig)


st.markdown("""----""")
st.write("3D Scatter Plot of Clusters")
fig = px.scatter_3d(PCA_df_2_scaled,x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"],
                    z=PCA_df_2_scaled["column_3"],color='Cluster')
st.plotly_chart(fig)





st.markdown("""----""")
if st.checkbox("Show Codes and comments"):
  
    # display the code in page
    st.markdown('---')
    st.subheader('Code')
    body ="""
        import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

np.random.seed(42)

df = pd.read_csv("marketing_campaign.csv", sep="\t")

df.head()

df.info()

df.describe()

msno.matrix(df)

df = df.dropna()

len(df)

msno.matrix(df)

df['Dt_Customer'] = df['Dt_Customer'].astype('datetime64[ns]')

# unique values present in each column
df.nunique()

print("time of inserting first Customer in database",df["Dt_Customer"].min())
print("time of inserting last customer in database", df["Dt_Customer"].max())

Membership_Period = []
for i in df["Dt_Customer"]:
    j = df["Dt_Customer"].max() - i
    Membership_Period.append(j)

df['Membership_Period'] = Membership_Period
df['Membership_Period'] = pd.to_numeric(df['Membership_Period'].dt.days, downcast='integer')

df.info()

df["Marital_Status"].value_counts()

df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'Couple')
df['Marital_Status'] = df['Marital_Status'].replace(['Single','Divorced','Widow','Alone','Absurd','YOLO'],'Single')
df["Marital_Status"].value_counts()

df["Education"].value_counts()

df['Education'] = df['Education'].replace(['PhD','Master'],'Postgraduate')
df['Education'] = df['Education'].replace(['Graduation'],'Graduate')
df['Education'] = df['Education'].replace(['2n Cycle','Basic'],'UnderGraduate')
df["Education"].value_counts()

df.info()

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

df.head()

fig = px.histogram(df, x="Age",color="Marital_Status")
fig.show()

To_Plot = [ "Income", "Recency", "Education", "Age", "Total_Spent", "Family_Size","Membership_Period","Family"]

g = sns.PairGrid(df[To_Plot], hue="Family")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot, size=df["Family_Size"])
g.add_legend()



df = df[(df["Age"]<90)]
df = df[(df["Income"]<600000)]

fig = px.scatter(df, x="Income", y="Total_Spent",
	         size="Family_Size", color="Education")
fig.show()



c = df.select_dtypes(include='object').columns
c

labelencoder = LabelEncoder()
df['Education'] = labelencoder.fit_transform(df['Education'])
df['Marital_Status'] = labelencoder.fit_transform(df['Marital_Status'])

df.info()

df.head(15)

df_2 = df.drop(columns=['Kidhome', 'Teenhome','AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain','NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases'])

df_2.info()

df_2.columns

scaler = StandardScaler()
scaler.fit(df_2)
df_2_scaled = pd.DataFrame(scaler.transform(df_2),columns= df_2.columns )
df_2_scaled.info()

df_2_scaled.head()

corrmat= df_2_scaled.corr()
plt.figure(figsize=(20,15))  
sns.heatmap(corrmat,annot=True)

pca = PCA(n_components=3)
pca.fit(df_2_scaled)
PCA_df_2_scaled = pd.DataFrame(pca.transform(df_2_scaled), columns=(["column_1","column_2", "column_3"]))
PCA_df_2_scaled.describe().T

fig = px.scatter_3d(PCA_df_2_scaled, x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"], z=PCA_df_2_scaled["column_3"],)
fig.show()

elbow = KElbowVisualizer(KMeans(), k=8, random_state=42)
elbow.fit(PCA_df_2_scaled)
elbow.show()
plt.show()

kmeancluster = KMeans(n_clusters=4,random_state=42)
PCA_df_2_scaled["Cluster"] = kmeancluster.fit_predict(PCA_df_2_scaled)

sns.countplot(data=PCA_df_2_scaled, x="Cluster", palette="tab20")
plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.show()

pl = sns.scatterplot(data = PCA_df_2_scaled,x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"],hue=PCA_df_2_scaled["Cluster"], palette="tab20")
pl.set_title("Cluster")
plt.legend()
plt.show()

print(PCA_df_2_scaled.shape)

PCA_df_2_scaled.head()

labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

cluster1_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==0].shape[0]
cluster2_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==1].shape[0]
cluster3_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==2].shape[0]
cluster4_val = PCA_df_2_scaled[PCA_df_2_scaled["Cluster"]==3].shape[0]
values = [cluster1_val, cluster2_val, cluster3_val, cluster4_val]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole = 0.5, title="Clusters")])
fig.show()

fig = px.scatter_3d(PCA_df_2_scaled,x=PCA_df_2_scaled["column_1"], y=PCA_df_2_scaled["column_2"],
                    z=PCA_df_2_scaled["column_3"],color='Cluster')
fig.show()


    """
    st.code(body,language='python')





if st.checkbox("Report of Clusters"):
    st.info("""
    This is a report created with pandas profiling, it will show you the dataframe's structure,
    Please Wait...
    """)

    pr = ProfileReport(PCA_df_2_scaled, title="Pandas Profiling Report")

    st_profile_report(pr)

    

