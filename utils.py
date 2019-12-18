''' Common utility funtions '''
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stopwords(text):
    '''Remove stop words'''
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

def lemmatize_stemming(text):
    '''Lemmetize and pos tagging'''
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def stemming(text):
    '''Stemming'''
    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)


