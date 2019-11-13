import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
lmtzr = WordNetLemmatizer()
stop = stopwords.words('english')

ps = PorterStemmer()

df = pd.read_csv("test.csv", skipinitialspace=True, usecols=[0, 1])

#data cleansing, stripwhitespace, replace punct, replace numbers, tolower
df['Comments'] = df['Comments'].str.strip()
df['Comments'] = df['Comments'].str.replace('[^\w\s]', '')
df['Comments'] = df['Comments'].str.lower()
df['Comments'] = df['Comments'].str.replace('\d+', '')

#lemmatize
df['Comments'] = [lmtzr.lemmatize(x) for x in df['Comments']]

#stem
df['Comments'] = df['Comments'].apply(word_tokenize)
df['Comments'] = df['Comments'].apply(lambda x: ' '.join([ps.stem(y) for y in x]))

# stopwords using english
df['Comments'] = df['Comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# create ngrams
word_vectorizer = CountVectorizer(ngram_range=(2,3), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df['Comments'])

#running agg count of each gram
frequencies = sum(sparse_matrix).toarray()[0]

#store dataframe in new variable
p = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

#sort frequency by high to low
p.sort_values(by=['frequency'], inplace=True, ascending=False)

#print/store or store into csv file
print(p)