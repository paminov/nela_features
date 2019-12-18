'''Frequency Heuristic Feature class definition'''
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


class FrequencyHeuristicFeature(object):
    #nela_score_values_reduced_ds_with_frq_cnt.csv: https://drive.google.com/open?id=1--ixtPMdoUIaYXBNPlUwWsQJkLWcndBy

    __datasets = {
                'data': ('1--ixtPMdoUIaYXBNPlUwWsQJkLWcndBy', 'nela_score_values_reduced_ds_with_frq_cnt.csv')
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
        self.data = self.data.round({"aggregated": 0})
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None,
									 strip_accents='unicode', analyzer='word',
									 token_pattern=r'\w{1,}', ngram_range=(1,5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1)
        self.X_train, self.X_test, self.y_train, self.y_test = \
					train_test_split(self.data[['source_cnt', 'name']],
									 self.data['aggregated'],
									 test_size=0.25, random_state=42)
        self.X_train_text = self.tfidf.fit_transform(self.X_train['name'])
        self.X_test_text = self.tfidf.transform(self.X_test['name'])
        self.X_train_val = self.X_train.drop(['name'], axis=1).values
        self.X_test_val = self.X_test.drop(['name'], axis=1).values
        self.X_train = sparse.hstack([self.X_train_val, self.X_train_text]).tocsr()
        self.X_test = sparse.hstack([self.X_test_val, self.X_test_text]).tocsr()

    @staticmethod
    def __text_preprocess(df):
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
        #remove less than 3 letter words
        df['name']    = df.name.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
        return df[['name', 'source_cnt', 'aggregated']]

    def __vectorize(self, df):
        x_text = self.tfidf.transform(df['name'])
        x_val = df.drop(['aggregated','name'], axis=1).values
        x = sparse.hstack([x_val, x_text]).tocsr()
        y = df['aggregated'].values
        return x, y

    def __train_multinomial_bayes(self):
        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        return nb

    def predict(self, name, source_cnt, aggregated):
        model = self.__train_multinomial_bayes()
        df = pd.DataFrame(data={"name": [name],
                   		        "source_cnt": [source_cnt],
                         		"aggregated": [aggregated]})
        df = self.__text_preprocess(df)
        x, y = self.__vectorize(df)
        prediction = model.predict(x)
        return prediction[0]/5.0
