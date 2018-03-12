import nltk
import pandas as pd
import numpy as np
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
import time
import subprocess
import requests
import re
subprocess.Popen('C:\\Users\\Pranav\\Anaconda3\\elasticsearch-6.2.2\\bin\\elasticsearch.bat')
time.sleep(10)
res = requests.get('http://localhost:9200')
print(res.content)
from elasticsearch import Elasticsearch
es = Elasticsearch()

## Keyword Extraction from Questions ## 


# q = "Which companies went bankrupt in MONTH of YEAR?"
# q = "What affects GDP?"
# q = "What percentage of drop or increase is associated with Z?"
# q = "Who is the CEO of COMPANYNAME?"
q = input("Please state your question: ")

r = Rake()
r.extract_keywords_from_text(q)
q_r = r.get_ranked_phrases()

if 'bankrupt' in q:
    q_tok = word_tokenize(q_r[0])
    q_tok.remove('companies')
    q_tok.remove('went')
    q_r = q_r + q_tok + ['declare']
    del q_r[0]
    qtype = 1
elif 'GDP' in q:
    q_r = word_tokenize(q_r[0])
    q_r = q_r + ['grow'] + ['decline']
    qtype = 2
elif 'percentage' in q:
    qtype = 3
    q_r.remove('associated')
    q_r.remove('increase')
    q_r.remove('drop')
    q_r = q_r + ['GDP'] + ['%']
elif 'CEO' in q:
    qtype = 4


## PREPROCESSING ##


# Normalization of Keywords
keywords = [w.lower() for w in q_r]

# Stemming of Keywords
st = LancasterStemmer()
keywords = [st.stem(w) for w in keywords]

# Lemmatization of Keywords
lm = WordNetLemmatizer()
keywords = [lm.lemmatize(w) for w in keywords]

# Read in Corpus
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_QA\\text_data\\*.txt"
files = glob.glob(PATH)
doc_list = list()
for name in files:
    with open(name, 'r',errors = "ignore",encoding = "utf-8") as test_data:
        data=test_data.read().replace('\n', '')
    doc_list.append(data)
corpus = ' '.join(doc_list)

# Remove Punctuation from Corpus
punc = (",`~{}|:./;'?&-$()[]+_=-:*^\<>#@&")
doc_list1 = [''.join(c for c in s if c not in punc) for s in doc_list]

# Normalization of Corpus
doc_list1 = [doc.lower() for doc in doc_list1]

# Stemming of Corpus
pp_docs = list()
for doc in doc_list1:
    temp = word_tokenize(doc)
    temp = [st.stem(w) for w in temp]
    doc = ' '.join(temp)
    pp_docs.append(doc)
doc_list1 = pp_docs.copy()

# Lemmatization of Corpus
pp_docs = list()
for doc in doc_list1:
    temp = word_tokenize(doc)
    temp = [lm.lemmatize(w) for w in temp]
    doc = ' '.join(temp)
    pp_docs.append(doc)

pp_corp = ' '.join(pp_docs)
pp_list = list()
for doc in pp_docs:
    pp_list.append(word_tokenize(doc))


## Use Elasticsearch ## 


# Find Relevant Documents
es.indices.delete(index = "documents")
for ii in range(0,len(pp_docs)-1):
    test = es.index(index = "documents",doc_type = "list",id = ii, body = {"token_list":pp_list[ii]})

ranked_docs = [0] * 731
for word in keywords:
    results = es.search(index = "documents", filter_path = ['hits.hits._id','hits.hits._score'], q = word,size = 731)
    matches = results['hits']['hits']
    for entry in matches:
        ranked_docs[int(entry['_id'])] = ranked_docs[int(entry['_id'])] + entry['_score']
        
# Select Highly Ranked Documents
ranked_docs = pd.DataFrame(np.array(ranked_docs).reshape(len(ranked_docs),1),columns = ["score"])
ranked_docs = ranked_docs.sort_values(by = 'score',ascending = False)
best_docs = ranked_docs.index.values[0:20]

# Sentence Tokenize
final_docs = list()
for doc in best_docs:
    final_docs.append(doc_list[doc])
final_corp = ' '.join(final_docs)

final_sent = sent_tokenize(final_corp)
sent_list = list()
for sent in final_sent:
    sent_list.append(word_tokenize(sent))

# Extract Question Keywords Again Without Preprocessing Tokens
stop_words = set(stopwords.words('english'))
stop_words.update(['?','Who','What','Which','went'])
q_r = word_tokenize(q)
keywords = list()
for word in q_r:
    if word not in stop_words:
        keywords.append(word)
if qtype == 1:
    keywords.remove('companies')
    keywords = keywords + ['declare']
elif qtype == 2:
    keywords = keywords + ['grow'] + ['decline']
elif qtype == 3:
    keywords.remove('associated')
    keywords = keywords + ['grow'] + ['GDP'] + ['decline'] + ['%']

# Use Elasticsearch to Find Relevant Sentences
es.indices.delete(index = "sentences")
for ii in range(0,len(final_sent)-1):
    test = es.index(index = "sentences",doc_type = "sentences",id = ii, body = {"sent_list":sent_list[ii]})

ranked_sents = [0] * len(sent_list)
for word in keywords:
    results = es.search(index = "sentences", filter_path = ['hits.hits._id','hits.hits._score'],q = word, size = 10000)
    matches = results['hits']['hits']
    for entry in matches:
        ranked_sents[int(entry['_id'])] = ranked_sents[int(entry['_id'])] + entry['_score']
        
# Select Highly Ranked Sentences
ranked_sents = pd.DataFrame(np.array(ranked_sents).reshape(len(ranked_sents),1),columns = ["score"])
ranked_sents = ranked_sents.sort_values(by = 'score',ascending = False)
best_sents = ranked_sents.index.values[0:10]

sentences = list()
for sent in best_sents:
    sentences.append(final_sent[sent])


## Extract Answers ##


if qtype == 1:
    # Use Regular Expressions to Find Company Names
    results = list()
    for sent in sentences:
        if 'bankrupt' in sent:
            results.append(sent)
    results = ' '.join(results)
    answers1 = re.findall('[A-Z][a-z]+\s[A-Z][a-z]+',results)
    answers2 = re.findall('[A-Z][A-Z]+',results)
    answers3 = re.findall('[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+',results)
    answers4 = re.findall('[A-Z][a-z]+',results)
    answer = answers1 + answers2 + answers3 + answers4
    print(answer)
elif qtype == 2:
    # Use POS Tagging to Find all Nouns in Important Sentences
    results = list()
    for sent in sentences:
        if 'GDP' in sent:
            results.append(sent)
    results = ' '.join(results)
    words = word_tokenize(results)
    pos_words = nltk.pos_tag(words)
    answer1 = list()
    for entry in pos_words:
        if entry[1] == "NN":
            answer1.append(entry[0])
    answer = list()
    for entry in answer1:
        if entry != "%":
            answer.append(entry)
    print(answer)
elif qtype == 3:
    # Use Regular Expressions to Find Relevant Percentages
    results = list()
    for sent in sentences:
        if 'GDP' in sent:
            results.append(sent)
    results = ' '.join(results)
    p1 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+%[\s\)]?',results)
    p2 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercent[\s\)]?',results)
    p3 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercentile\spoints?[\s\)]?',results)
    p4 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercentage\spoints?[\s\)]?',results)
    p5 = re.findall('\s[\)\~]?[A-Za-z]+\spercent[\s\)]?',results)
    p6 = re.findall('\s[\)\~]?[A-Za-z]+\spercentage\spoints?[\s\)]?',results)
    p7 = re.findall('\s[\)\~]?[A-Za-z]+\spercentile\spoints?[\s\)]?',results)
    percents = p1 + p2 + p3 + p4 + p5 + p6 + p7
    answer = list()
    for pct in percents:
        answer.append(pct[1:-1])
    print(answer[0])
elif qtype == 4:
    # Find Most Common Term with Regular Expressions
    sentences = ' '.join(sentences)
    ceo = re.findall('CEO\s([A-Z][a-z]+\s[A-Z][a-z]+)',sentences)
    def most_common(lst):
        return max(set(lst), key=lst.count)
    answer = most_common(ceo)
    print(answer)
