import pandas as pd
import numpy as np
import random
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

avocado_df=pd.read_csv('avocado.csv')
avocado_df.head()
avocado_df.tail(10)
avocado_df.describe()
avocado_df.info()
avocado_df.isnull().sum()

#explore dataset
avocado_df=avocado_df.sort_values('Date')
plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'],avocado_df['AveragePrice'])


#distribution of average price
plt.figure(figsize=(10,6))
sns.distplot(avocado_df['AveragePrice'],color='b')

sns.violinplot(y='AveragePrice',x='type',data=avocado_df)

#bar chart to indicate number of regions
sns.set(font_scae=0.7)
plt.figure(figsize=[25,12])
sns.countplot(x='region',data=avocado_df)
plt.xticks(rotation=45)

#bar chart to indicate count in every year
sns.set(font_scae=0.7)
plt.figure(figsize=[25,12])
sns.countplot(x='year',data=avocado_df)
plt.xticks(rotation=45)


#plot the avocado prices vs regions for conventional avocados
conventional=sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='conventional'],
                         hue='year',height=20)

organic=sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='organic'],
                         hue='year',height=20)





#prepare data before feeding to prophet
#we require only price and date
avocado_prophet_df=avocado_df[['Date','AveragePrice']]
avocado_prophet_df=avocado_prophet_df.rename(columns={'Date':'ds','AveragePrice':'y'})
avocado_prophet_df


#main work
m=Prophet()
m.fit(avocado_prophet_df)

future=m.make_future_dataframe(periods=365)
forecast=predict(future)
forecast
figure=m.plot(forcast,xlabel='Date',ylabel='Price')
figure2=plot_components(forecast)

#now developing region specific moddel
avocado_df=pd.read_csv('avocado.csv')
avocado_df_sample=avocado_df[avocado_df['region']=='West']
avocado_df_sample=avocado_df_sample.sort_values('Date')
plt.plot(avocado_df_sample['Date'],avocado_df_sample['AveragePrice'])
avocado_df_sample=avocado_df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})

m=Prophet()
m.fit(avocado_df_sample)

future=m.make_future_dataframe(periods=365)
forecast=predict(future)

figure=m.plot(forcast,xlabel='Date',ylabel='Price')
figure2=plot_components(forecast)
