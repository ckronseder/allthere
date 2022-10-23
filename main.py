# Recommendation engine based on similar investors. The idea is that similar investors buy into similar products

# Uses streamlit for a simplified UI and UX

import streamlit as st
import numpy as np
import pandas as pd
import nltk
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

def prepare_data():
    funds = pd.read_csv("funds_list.csv")
    ratings = pd.read_csv("ratings.csv")
    dataset = funds.merge(ratings)
    dataset = dataset.loc[:, ["userId", "movieId", "title", "genres", "rating"]]
    df_ratings = dataset.loc[:, ["title", "rating"]].groupby("title").mean()
    genres = dataset["genres"]
    portfolios = pd.read_csv('client_portfolio.csv')

    return [dataset, genres, df_ratings, portfolios]

def portfolio(dataframe, clientID):
    """ Returns a dataframe of a particular client. The portfolio is truncated at 20 entries"""
    select_client = dataframe[dataframe['userId']==clientID]
    select_client.reset_index()

    if select_client.shape[0] > 50:
        select_client = select_client.head(50)

    return select_client


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    st.image('Logo_ALLindice_H.gif')
    st.title('ALLTHERE')
    st.write('Provides a list of funds, depending on the investment portfolio of a client')
    initial_data = prepare_data()

    lemmatizer = WordNetLemmatizer()
    li = []
    for i in range(len(initial_data[1])):
        temp = initial_data[1][i].split("|")
        for j in range(len(temp)):
            temp[j] = lemmatizer.lemmatize(temp[j])
        li.append(" ".join(temp))

    cv = CountVectorizer()
    X = cv.fit_transform(li).toarray()

    genres = pd.DataFrame(X, columns=cv.get_feature_names_out())
    dataset = initial_data[0].iloc[:,:-2]
    new_dataset = dataset.join(genres)

    users = new_dataset.drop(['movieId', 'title'], axis=1)
    user_fundmat = users.groupby('userId').sum()
    X = user_fundmat.iloc[:,:].values

    classifier = NearestNeighbors()
    classifier.fit(X)

    #uid = int(input("Enter User Id "))
    uid = int(st.number_input('Insert Client ID (1 - 500)', value=1))
    if uid:

        li = classifier.kneighbors([X[uid - 1]], n_neighbors=5, return_distance=False)
        current_user = new_dataset.loc[new_dataset["userId"] == li[0][0], :]["title"].values
        similar_user = new_dataset.loc[new_dataset["userId"] == li[0][1], :]["title"].values
        fund_list = [fund for fund in similar_user if fund not in current_user]
        for i in range(len(fund_list)):
            fund_list[i] = (fund_list[i], initial_data[2]['rating'][initial_data[2].index == fund_list[i]].values[0])

        df = pd.DataFrame(columns=['Fund', 'Fund Rating'], data=fund_list)
        st.write('Potential funds:')
        st.table(df.head(5))

        st.write("Existing portfolio of client", uid)
        st.table(portfolio(initial_data[3], uid)['fund'])



