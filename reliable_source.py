'''Reliable Source Feature class definition'''
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from .utils import *


class ReliableSourceFeature(object):
    #nela_score_values_reduced_ds.csv: https://drive.google.com/open?id=1_LyNVDK0bJKEQKkYlUIpLay4bHAt5UmM

    __datasets = {
                'data': ('1-MdGo8vMMLWMZBcb_K0dKz56FzNEra7W', 'nela_score_values_reduced_ds.csv')
            }

    def __init__(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
        self.__load_datasets()
        self.stop_words = None
        self.headline_words = None

    def __load_dataset_from_gdrive(self, file_id, file_name):
        downloaded = self.drive.CreateFile({'id':file_id})
        downloaded.GetContentFile(file_name)
        return pd.read_csv(file_name)

    def __load_datasets(self):
        data = self.__load_dataset_from_gdrive(*self.__datasets['data'])
        self.data = self.__text_preprocess(data).dropna()
        self.data.dropna(how='any',axis=0)
        self.data = self.data.round({"aggregated": 0})

        self.tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1, lowercase=False)
        self.tfidf1 = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                      analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                                      use_idf=1, smooth_idf=1, sublinear_tf=1, lowercase=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[['source', 'name']], self.data['aggregated'], test_size=0.25, random_state=42)
        self.X_train_source = self.tfidf.fit_transform(self.X_train['source'])
        self.X_test_source    = self.tfidf.transform(self.X_test['source'])
        self.X_train_name = self.tfidf1.fit_transform(self.X_train['name'])
        self.X_test_name    = self.tfidf1.transform(self.X_test['name'])
        self.X_train = sparse.hstack([self.X_train_source, self.X_train_name]).tocsr()
        self.X_test    = sparse.hstack([self.X_test_source, self.X_test_name]).tocsr()

    @staticmethod
    def __text_preprocess(df):
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
        return df[['source', 'name', 'aggregated']]

    def __vectorize(self, df):
        x_source = self.tfidf.transform(df['source'])
        x_name = self.tfidf1.transform(df['name'])
        x = sparse.hstack([x_source, x_name]).tocsr()
        y = df['aggregated'].values
        return x, y

    def __train_multinomial_bayes(self):
        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        return nb

    def predict(self, source, name, aggregated):
        model = self.__train_multinomial_bayes()
        df = pd.DataFrame(data={"source": [source],
                                "name": [name],
                                "aggregated": [aggregated]})
        df = self.__text_preprocess(df)
        x, y = self.__vectorize(df)
        prediction = model.predict(x)
        return prediction[0]/5.0
