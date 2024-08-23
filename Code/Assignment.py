import nltk
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter

count = 1
## Run the next line of code only once 
nltk.download('all')

# This function returns the most frequent Percent% N-gram from the given tokens 
def MostFrequentNGrams(Word_Token, N, Percent, Sum):
    TokenI = ngrams(Word_Token, N)
    Ngram_tmp = Counter(TokenI)
    Ngram_temp = Ngram_tmp.most_common(len(Ngram_tmp))
    
    Ngram_reqd = math.ceil(Percent*Sum)
    Ngram_dict = dict()
    Ngram_count = 0
    i = 0
    while Ngram_count <= Ngram_reqd:
        if N>1:
            Ngram_dict[Ngram_temp[i][0]] = Ngram_temp[i][1]
        else:
            Ngram_dict[Ngram_temp[i][0][0]] = Ngram_temp[i][1]
        Ngram_count = Ngram_count + Ngram_temp[i][1]
        i = i+1
    return Ngram_dict

# This function plots the graph of the log(rank) vs log(frequency) and compares it with the Zipfian distribution 
def PlotGraph(diction):
    global count
    title = ''
    if count < 4:
      title = 'Complete '
    elif count < 7:
      title = 'Stemmed '
    else:
      title = 'Lemmatized '
    
    if count%3 == 1:
      title = title + 'Unigram Analysis: '
    elif count%3 == 2:
      title = title + 'Bigram Analysis: '
    else:
      title = title + 'Trigram Anaysis: '

    dictionary = {}
    i = 1
    for key in diction:
      dictionary[math.log(i)] = math.log(diction[key])
      i = i+1
    dataFrame = pd.DataFrame(list(dictionary.items()), columns=['Word','Frequency'])
    x = dataFrame['Word']
    y = dataFrame['Frequency']
    plt.plot(x, y, '-b', label = 'Actual Distribution of log-log graph')
    plt.title(title + 'Log of Frequency vs Log of Rank')
    plt.xlabel('Log(Word Rank)')
    plt.ylabel('Log(Frequency)')
    X = np.linspace(0, dictionary[0], 100)
    Y = dictionary[0] - X
    plt.plot(X, Y, '-r', label = 'Prediction from Zipf\'s Law')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.show()
    plt.savefig('Graph_'+str(count)+'.png')
    count = count + 1


# Reading the file
f = open("wiki_69.txt", "r", encoding='utf-8')
raw_text = f.read(-1)
f.close()

# Parsing the given HTML text using Beautiful Soup and tokenizing using word_tokenize
raw_text = bs(raw_text, 'html.parser').get_text()
word_token = nltk.word_tokenize(raw_text)

# Removing punctuations marks (Optional)
word_tokens = []
punctuation = ['.', ',', '(', ')', '"', '!', ':', '-']
for word in word_token:
    if word not in punctuation:
        word_tokens.append(word)

# Converting the word_tokens to unigram, bigram and trigram
unigrams =  ngrams(word_tokens, 1)
bigrams =  ngrams(word_tokens, 2)
trigrams =  ngrams(word_tokens, 3)
unigram = Counter(unigrams)
bigram = Counter(bigrams)
trigram = Counter(trigrams)

#####################################    UNIGRAM    ####################################
unigram_dictionary = MostFrequentNGrams(word_tokens, 1, 0.9, sum(unigram.values()))

#####################################    BIGRAM    ####################################    
bigram_dictionary = MostFrequentNGrams(word_tokens, 2, 0.8, sum(bigram.values()))
    
#####################################    TRIGRAM    ####################################   
trigram_dictionary = MostFrequentNGrams(word_tokens, 3, 0.7, sum(trigram.values()))
    
#####################################    STEMMING    #################################### 
Stemmed_Tokens = []
PS = PorterStemmer()
for word in word_tokens:
    Stemmed_Tokens.append(PS.stem(word))

stemmed_unigrams =  ngrams(Stemmed_Tokens, 1)
stemmed_bigrams =  ngrams(Stemmed_Tokens, 2)
stemmed_trigrams =  ngrams(Stemmed_Tokens, 3)
stemmed_unigram = Counter(stemmed_unigrams)
stemmed_bigram = Counter(stemmed_bigrams)
stemmed_trigram = Counter(stemmed_trigrams)

        #####################################    UNIGRAM    ####################################
stemmed_unigram_dictionary = MostFrequentNGrams(Stemmed_Tokens, 1, 0.9, sum(stemmed_unigram.values()))

        #####################################    BIGRAM    ####################################    
stemmed_bigram_dictionary = MostFrequentNGrams(Stemmed_Tokens, 2, 0.8, sum(stemmed_bigram.values()))
    
        #####################################    TRIGRAM    ####################################    
stemmed_trigram_dictionary = MostFrequentNGrams(Stemmed_Tokens, 3, 0.7, sum(stemmed_trigram.values()))
  

#####################################    LEMMATIZING    ####################################
Lemmed_Tokens = []
lem = WordNetLemmatizer()
for word in word_tokens:
    Lemmed_Tokens.append(lem.lemmatize(word))

lemmed_unigrams =  ngrams(Lemmed_Tokens, 1)
lemmed_bigrams =  ngrams(Lemmed_Tokens, 2)
lemmed_trigrams =  ngrams(Lemmed_Tokens, 3)
lemmed_unigram = Counter(lemmed_unigrams)
lemmed_bigram = Counter(lemmed_bigrams)
lemmed_trigram = Counter(lemmed_trigrams)

        #####################################    UNIGRAM    ####################################
lemmed_unigram_dictionary = MostFrequentNGrams(Lemmed_Tokens, 1, 0.9, sum(lemmed_unigram.values()))

        #####################################    BIGRAM    ####################################    
lemmed_bigram_dictionary = MostFrequentNGrams(Lemmed_Tokens, 2, 0.8, sum(lemmed_bigram.values()))
    
        #####################################    TRIGRAM    ####################################    
lemmed_trigram_dictionary = MostFrequentNGrams(Lemmed_Tokens, 3, 0.7, sum(lemmed_trigram.values()))

###################################   DISPLAYING THE RESULTS    ####################################
print('The total number of Unigrams present are : ' + str(len(unigram)))
print('The number of Unigrams required to cover 90% of corpus is = ' + str(len(unigram_dictionary)))
PlotGraph(unigram_dictionary)
print("\n\n"+'The total number of Bigrams present are : ' + str(len(bigram)))
print('The number of Bigrams required to cover 80% of corpus is = ' + str(len(bigram_dictionary)))
PlotGraph(bigram_dictionary)
print("\n\n"+'The total number of Trigrams present are : ' + str(len(trigram))) 
print('The number of Trigrams required to cover 70% of corpus is = ' + str(len(trigram_dictionary)))
PlotGraph(trigram_dictionary)  

print("\n\n"+'The total number of Stemmed Unigrams present are : ' + str(len(stemmed_unigram))) 
print('The number of Stemmed Unigrams required to cover 90% of corpus is = ' + str(len(stemmed_unigram_dictionary)))
PlotGraph(stemmed_unigram_dictionary)
print("\n\n"+'The total number of Stemmed Bigrams present are : ' + str(len(stemmed_bigram)))
print('The number of Stemmed Bigrams required to cover 80% of corpus is = ' + str(len(stemmed_bigram_dictionary)))
PlotGraph(stemmed_bigram_dictionary)
print("\n\n"+'The total number of Stemmed Trigrams present are : ' + str(len(stemmed_trigram)))
print('The number of Stemmed Trigrams required to cover 70% of corpus is = ' + str(len(stemmed_trigram_dictionary)))
PlotGraph(stemmed_trigram_dictionary)  

print("\n\n"+'The total number of Lemmatized Unigrams present are : ' + str(len(lemmed_unigram))) 
print('The number of Lemmatized Unigrams required to cover 90% of corpus is = ' + str(len(lemmed_unigram_dictionary)))
PlotGraph(lemmed_unigram_dictionary) 
print("\n\n"+'The total number of Lemmatized Bigrams present are : ' + str(len(lemmed_bigram)))
print('The number of Lemmatized Bigrams required to cover 80% of corpus is = ' + str(len(lemmed_bigram_dictionary)))
PlotGraph(lemmed_bigram_dictionary)
print("\n\n"+'The total number of Lemmatized Trigrams present are : ' + str(len(lemmed_trigram)))
print('The number of Lemmatized Trigrams required to cover 70% of corpus is = ' + str(len(lemmed_trigram_dictionary)))
PlotGraph(lemmed_trigram_dictionary)

##############################   CHI-SQUARE TEST ON BIGRAMS    ####################################
totalCount = sum(bigram.values())
chi_bigram_dict = dict()
for key in bigram:
  one = unigram[tuple([key[0]])]
  two = unigram[tuple([key[1]])]
  both = bigram[key]
  oneOnly = one - both
  twoOnly = two - both
  neither = totalCount - oneOnly - twoOnly - both
  
  p_one = one / totalCount
  p_two = two / totalCount

  Ex_both = totalCount*p_one*p_two
  Ex_oneOnly = totalCount*p_one*(1-p_two)
  Ex_twoOnly = totalCount*(1-p_one)*p_two
  Ex_neither = totalCount*(1-p_one)*(1-p_two)

  C2 = ((both - Ex_both)**2)/Ex_both + ((oneOnly - Ex_oneOnly)**2)/Ex_oneOnly + ((twoOnly - Ex_twoOnly)**2)/Ex_twoOnly + ((neither - Ex_neither)**2)/Ex_neither
  chi_bigram_dict[key] = C2

chisqr_bigram = dict(sorted(chi_bigram_dict.items(), key=lambda x:(-x[1], x[0])))
##############################   DISPLAYING TOP 20 COLLOCATIONS    ####################################
print('\n\nTHE TOP 20 BI-GRAM COLLOCATIONS IN THE TEXT CORPUS ARE :-')
count_chi = 1
for key in chisqr_bigram:
  if count_chi > 20:
    break
  print(str(count_chi)+') '+key[0][1:] +' '+key[1])
  count_chi += 1
