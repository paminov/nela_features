'''Post and Social Media Activities feature class definition'''
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


class SocialMediaActivitiesFeature(object):
    #NELA_social_features.csv: https://drive.google.com/open?id=1-2H3wbOCKTTV_zv2Ez2OhoC9k8FfdLOF
    #aggregated_labels.csv: https://drive.google.com/open?id=1e_H3EaaSoupdnMTL132s6YyrwDvSR_Ib

    __datasets = {
                'data': ('1-2H3wbOCKTTV_zv2Ez2OhoC9k8FfdLOF', 'NELA_social_features.csv'),
                'labels': ('1e_H3EaaSoupdnMTL132s6YyrwDvSR_Ib', 'aggregated_labels.csv')
            }

    def __init__(self):
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
        labels = self.__load_dataset_from_gdrive(*self.__datasets['labels'])
        self.data, self.labels = data, labels
        self.data_all = self.__aggregate_datasets(data, labels)
        self.data_all = self.__text_preprocess(self.data_all).dropna()
        self.data_all = self.data_all.round({"aggregated": 0})
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_all[['source','Facebook Reaction Count','Facebook Share Count','Facebook Comment Count']], self.data_all['aggregated'], test_size=0.25, random_state=42)
        self.X_train_source = self.tfidf.fit_transform(self.X_train['source'])
        self.X_test_source    = self.tfidf.transform(self.X_test['source'])
        self.x_train_val = self.X_train.drop(['source'], axis=1).values
        self.x_test_val = self.X_test.drop(['source'], axis=1).values
        self.X_train = sparse.hstack([self.x_train_val, self.X_train_source]).tocsr()
        self.X_test    = sparse.hstack([self.x_test_val, self.X_test_source]).tocsr()

    def __aggregate_datasets(self, data, labels):
        df = data.copy()
        dict={}
        for i in labels.index:
            key=labels['Source'][i]
            dict[key]=labels['aggregated'][i]

        aggregated_labels= []

        for i in df.index:
            aggregated_labels.append(dict.get(df['source'][i]))

        df['aggregated']=aggregated_labels
        return df

    def __train_model(self):
        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        return nb

    @staticmethod
    def __text_preprocess(df):
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

        return df[['Facebook Reaction Count', 'Facebook Share Count', 'Facebook Comment Count','source', 'aggregated']]

    def __vectorize(self, df):
        x_source = self.tfidf.transform(df['source'])
        x_val = df.drop(['aggregated','source'], axis=1).values
        x = sparse.hstack([x_source,x_val]).tocsr()
        y = df['aggregated'].values
        return x, y

    def predict(self, source, fbreaction, fbshare, fbcomment, aggregated, return_int=False):
        model = self.__train_model()
        df = pd.DataFrame(data={"source": [source],
                                'Facebook Reaction Count':[fbreaction],
                                'Facebook Share Count':[fbshare],
                                'Facebook Comment Count':[fbcomment],
                                'aggregated': [aggregated]})
        df = self.__text_preprocess(df)
        X, y = self.__vectorize(df)
        prediction = model.predict(X)
        return prediction[0]/5
