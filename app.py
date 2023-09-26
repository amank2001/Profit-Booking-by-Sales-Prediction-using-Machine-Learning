import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as pltpip
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error


st.title('Car Sales Prediction')


if __name__ == '__main__':
    uploaded_files = st.file_uploader("Upload your dataset",type=["csv"], accept_multiple_files=True)
    if uploaded_files is not None:
        for file in uploaded_files:
            file_contents = file.read()
            st.write(f"File name: {file.name}")


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#Describing Data
if st.button("Submit"):
    st.subheader('Car Sales Data of 79 weeks')
    st.write(df_train.describe())


    #String to Date Time Format
    df_train['date'] = pd.to_datetime(df_train['date'], format = '%d/%m/%Y')
    df_test['date'] = pd.to_datetime(df_test['date'], format = '%d/%m/%Y')


    #Visualizations
    st.subheader('Private Orders per Day vs Date Chart')
    fig = pltpip.figure(figsize=(12,6))
    ax = sns.lineplot(x="date", y="private_orders", data=df_train)
    ax.set_title('Private Orders per Day', fontsize=30)
    ax.set_xlabel('Date', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    pltpip.tight_layout()
    st.pyplot(fig)

    st.subheader('Private Orders per Day vs Week Chart')
    fig = pltpip.figure(figsize=(12,6))
    ax = sns.lineplot(x="week_id", y="private_orders", data=df_train)
    ax.set_title('Private Orders per day', fontsize=30)
    ax.set_xlabel('Week', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    pltpip.tight_layout()
    st.pyplot(fig)

    st.write('On studying the variation of number of private orders on both granularities: daily and weekly, It can be concluded that it has seasonality and noise. It does not seem to have some sort of trend.')

    st.subheader('Co-relation Matrix')
    ax = sns.heatmap(df_train.corr(), cmap='YlGnBu', annot_kws={'size': 20})
    st.pyplot(fig)


    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    #convert to an array
    values = df_train.iloc[:,2:].values
    #convert all columns to float
    #values.astype('float32')
    #normalize featues
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 7, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[range(56,63)], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values
    n_train_days = 7*64 
    train = values[:n_train_days, :]
    dev = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    dev_X, dev_y = dev[:, :-1], dev[:, -1]

    #Regressor Model
    xgb_regressor = xgb.XGBRegressor()
    xgb_regressor.fit(train_X, train_y)

    #prediction on training data
    training_data_prediction = xgb_regressor.predict(train_X)

    # R-squared value
    train_r2 = metrics.r2_score(train_y, training_data_prediction)

    st.subheader('Efficiency of our trained model')
    st.write('R-squared value = ', train_r2)

    def download_file():
        # Set the file path
        file_path = "csv_to_submit.csv"
        
        # Send the file as a response to the button click event
        with open(file_path, "rb") as f:
            data = f.read()
            st.download_button(
                label="Download your Predicted Result Dataset",
                data=data,
                file_name="Car Sales Prediction.csv",
                mime="text/plain",
            )

    if __name__ == '__main__':
        download_file()