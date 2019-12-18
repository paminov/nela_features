'''Social Credibility Feature class definition'''
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from .constants import *
from .utils import *
import numpy as np

class SocialCredibilityFeature(object):
    #nela_score_values_reduced_ds_with_frq_cnt.csv: https://drive.google.com/open?id=1_LyNVDK0bJKEQKkYlUIpLay4bHAt5UmM

    __datasets = {
                'data': ('1-MdGo8vMMLWMZBcb_K0dKz56FzNEra7W', 'nela_score_values_reduced_ds.csv')
            }

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
        self.__load_datasets()

    def __load_dataset_from_gdrive(self, file_id, file_name):
        downloaded = self.drive.CreateFile({'id':file_id})
        downloaded.GetContentFile(file_name)
        return pd.read_csv(file_name)

    def __load_datasets(self):
        data = self.__load_dataset_from_gdrive(*self.__datasets['data'])
        data = self.__amalgamate_twitter_data(data)
        data = data[np.isfinite(data['followers_count'])]
        self.data = self.__text_preprocess(data).dropna()
        self.data = self.data.round({"aggregated": 0})
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1, lowercase=False)
        self.tfidf1= TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1, lowercase=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[['source', 'name', 'ratio']], self.data['aggregated'], test_size=0.25, random_state=42)

        self.X_train_source = self.tfidf.fit_transform(self.X_train['source'])
        self.X_test_source    = self.tfidf.transform(self.X_test['source'])
        self.X_train_name = self.tfidf1.fit_transform(self.X_train['name'])
        self.X_test_name = self.tfidf1.transform(self.X_test['name'])
        self.X_train_val = self.X_train.drop(['source', 'name'], axis=1).values
        self.X_test_val    = self.X_test.drop(['source', 'name'], axis=1).values
        self.X_train = sparse.hstack([self.X_train_val, self.X_train_source, self.X_train_name]).tocsr()
        self.X_test    = sparse.hstack([self.X_test_val, self.X_test_source, self.X_test_name]).tocsr()

    def __text_preprocess(self, df):
        df = df[pd.notnull(df['name'])]
        df = df[pd.notnull(df['source'])]
        #convert to lower case
        df['source'] = df['source'].str.lower()
        #remove stop words
        df['source'] = df['source'].apply(remove_stopwords)
        #Lemmetize
        df['source'] = df['source'].apply(lemmatize_stemming)
        #stemming
        df['source'] = df['source'].apply(stemming)
        #remove punctuation
        df['source'] = df['source'].apply(remove_punctuation)
        #remove less than 3 letter words
        df['source']    = df.source.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))

        #convert to lower case
        df['name'] = df['name'].str.lower()
        #remove stop words
        df['name'] = df['name'].apply(remove_stopwords)
        #Lemmetize
        df['name'] = df['name'].apply(lemmatize_stemming)
        #stemming
        df['name'] = df['name'].apply(stemming)
        #remove punctuation
        df['name'] = df['name'].apply(remove_punctuation)

        df = self.__generate_ratio(df)

        return df[['source', 'name', 'ratio', 'aggregated' ]]

    def __amalgamate_twitter_data(self, df):
        df = df.copy()
        # create mapper
        map1 = {k:v[0] for k, v in sources_dictionary.items()}
        map2 = {k:v[1] for k, v in sources_dictionary.items()}
        map3 = {k:v[2] for k, v in sources_dictionary.items()}
        map4 = {k:v[3] for k, v in sources_dictionary.items()}
        # add twitter data per source
        df['followers_count'] = df['source'].map(map1)
        df['friends_count'] = df['source'].map(map2)
        df['listed_count'] = df['source'].map(map3)
        df['verified'] = df['source'].map(map4)
        return df

    def __generate_ratio(self, df):
        df = df.copy()
        for index, row in df.iterrows():
            followers_count = row['followers_count']
            friends_count = row['friends_count']
            ratio = friends_count / followers_count
            df.at[index, 'ratio'] = ratio
        return df

    def __vectorize(self, df):
        x_source = self.tfidf.transform(df['source'])
        x_name = self.tfidf1.transform(df['name'])
        x_val = df.drop(['aggregated','name','source'], axis=1).values
        x = sparse.hstack([x_val, x_source, x_name]).tocsr()
        y = df['aggregated'].values
        return x, y

    def __train_model(self):
        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        return nb

    def predict(self, source, name, followers_count, friends_count, aggregated):
        model = self.__train_model()
        df = pd.DataFrame(data={"source": [source],
                                "name": [name],
                                "followers_count": [followers_count],
                                "friends_count": [friends_count],
                                "aggregated": [aggregated]})
        df = self.__text_preprocess(df)
        X, y = self.__vectorize(df)
        prediction = model.predict(X)
        return prediction[0]/5.0
