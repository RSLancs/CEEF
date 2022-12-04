import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 


import glob
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Set, Optional
import random
import numpy as np 
import pickle 
import pandas as pd


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import accuracy_score


import xgboost, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers



##############################################################################
##...............import letters and re-structure into letter:recipient pairs....
def import_letters(option, in_data: Path, ) -> List[Tuple[str, str]]:
    files_in_folder_list = glob.glob(f'{in_data}/*1867_RAW.txt')
    
    letters = []  
    print('\n Letters read into python: \n')
    for file in files_in_folder_list: # iterate over all documents in input file
        print(file)
        
        file_name_in = os.path.basename(file)# collect file name

        with open(file, 'r') as f: 
            text_item = f.read() # read letter to memory
            letters.append([text_item, option])# append letter as nested list with recipient

    return(letters) # return all letters in input folder


## load in Margaret letter set
Margaret_text = import_letters('Margaret', './Margaret_letters_raw_text/')

## load in Joan letter set
joan_text = import_letters('Joan','./Joan_letters_raw_text/') 

##.....merge letter sets together
all_letters = Margaret_text + joan_text   



##################################################################################
##......having imported the letters, clean and standardise letters ............... 

wnl = WordNetLemmatizer() # define lemmatizer
tokenizer = RegexpTokenizer(r'\w+') # define tokenizer
stop_words = set(stopwords.words('english')) # create stop_word list


def clean_remove_stop_words(input_letters, remove_punct, lammatise, remove_stopwords): # iterate over all letters and:
    filtered_sentences =[] 

    print(f''' \n Processing steps carried out on each letter: \n\n Tokenisation = True \n Words to lower case = True \n punctuaion removed = {remove_punct} \n Lemmatisation = {lammatise} \n Stop words removed = {remove_stopwords} \n\n ''')

    for text in input_letters: # iterate over each letter and clean
        
        tokenise_words = nltk.word_tokenize(text[0]) # tokenise each letter
        
        lower_words = [word.lower() for word in tokenise_words] # lower all words
        
        if remove_punct == True:
            remove_punctu = [word for word in lower_words if word.isalpha()] # remove punctuation
        else:
            remove_punctu = lower_words

        if lammatise == True:
            lemma_words = [wnl.lemmatize(word) for word in remove_punctu] # get word lemmas
        else:
            lemma_words = remove_punctu

        if remove_stopwords == True:
            no_stop_words = [w for w in lemma_words if not w in stop_words]
        else:
            no_stop_words = lemma_words

        pro_joins = " ".join(no_stop_words)
        
        filtered_sentences.append([pro_joins, text[1]]) # return clear letter with recipient 
    

    print(f'First UNPROCESSED letter in letter set: \n{all_letters[0]} \n\n First PROCESSED letter in leter set: \n{filtered_sentences[0]} \n')

    ## convert letter:lebel pairs to dataframe
    all_letters_df = pd.DataFrame(filtered_sentences) # transform letters to dataframe
    all_letters_df.columns = ['text', 'label']
    
    print(f'All letters transformed to dataframe, with first row being: \n{all_letters_df.head(1)} \n')
    
    
    return(all_letters_df)



#################################################### 
##...... create a vector set for the complete letter set

def set_vectoriser_attib(input_data, vec_type, ngram):

    if vec_type == CountVectorizer:
        vectoriser = CountVectorizer(analyzer='word', token_pattern='\w+', preprocessor=None, lowercase=False, ngram_range=ngram, max_features=10000)
        vectoriser.fit(input_data['text'])
         
    if vec_type == TfidfVectorizer:

        vectoriser = TfidfVectorizer(analyzer='word', token_pattern='\w+', preprocessor=None, lowercase=False, ngram_range=ngram, max_features=10000)
        vectoriser.fit(input_data['text'])
        
    print(f'vectoriser type selected: \n{vectoriser} \n')
    
    print(f'the vocab is: \n{vectoriser.vocabulary_} \n')
    
    return vectoriser


########################################################################

'''
This function splits the data into train and test sets, where the test set is always ONE single letter. 
The function iteratively splits the data on all possible combinations and trains the model on each. This 
gives a truer idea of model accuracy and reduces the change of overfitting the model.
''' 

def ml_average(classifier, input_data, vectoriser):
    accuracy_list = []
    failed_predict = []

    ## transform input dataframe to nested list
    input_features = input_data.values.tolist()
    print(f'Length of full dataset is {len(input_features)} \n')
   

    ## iteratively split the data - tests set is one letter at a time - training set is the rest
    for i, letter in enumerate(input_features):
        
        # form training set and categories
        train_data = input_features[:i] + input_features[i+1:] 
        training_set = [t_set[0] for t_set in train_data]
        training_category = [t_set[1] for t_set in train_data]
        print(f'The length of the test data set is {len(train_data)}, and is missing letter: {i}')
        
        # form test set and category
        test_data = input_features[i:i+1]
        testing_set = [test_data[0][0]]
        testing_category = [test_data[0][1]]
        print(f'The length of the test data set is {len(test_data)}, and is letter: {i}')

        ###  encode category labels for training set 
        encoder = preprocessing.LabelEncoder()
        training_category = encoder.fit_transform(training_category)

        ## encode category labels for test set  
        cat_encode = {'Joan':0, 'Margaret':1}
        testing_category = [v for k, v in cat_encode.items() if k in testing_category]

        # trainsform training and test sets to vectorset 
        vec_train_set = vectoriser.transform(training_set)
        print(f'Shape of vectorised train set (length, N. features): \n{vec_train_set.shape}')
        vec_test_set = vectoriser.transform(testing_set)
        print(f'Shape of vectorised test set (length, N. features): \n{vec_test_set.shape}')

        # fit the training dataset on the NB classifier
        classifier.fit(vec_train_set, training_category)
        
        # predict the labels on validation dataset
        predictions_NB = classifier.predict(vec_test_set)
        
        # Use accuracy_score function to get the accuracy
        print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, testing_category)*100)
        print(f'Actual letter: {letter[1]}, prediction was: {[k for k, v in cat_encode.items() if v in predictions_NB]} \n')

        accuracy_list.append(accuracy_score(predictions_NB, testing_category)*100)
        
        if accuracy_score(predictions_NB, testing_category)*100 == 0.0:
            failed_predict.append([i, letter])

    return accuracy_list, failed_predict



#################################################
##....... run the script and set parameters......


### set process criteria of letters 
pro_all_letters = clean_remove_stop_words(all_letters, remove_stopwords=False, remove_punct=True, lammatise=True)


## vectorise the entire data set - vectoriser choices supported are TfidfVectorizer or CountVectorizer
trained_vectoriser = set_vectoriser_attib(pro_all_letters, vec_type=CountVectorizer, ngram=(1,2))


## train model and predict
model_predictions = ml_average(naive_bayes.MultinomialNB(), pro_all_letters, trained_vectoriser)
print(f'The results from each model is: \n{model_predictions[0]} \n\nNumber of predictors: {len(model_predictions[0])} \n\nThe models overall performance is: \n{sum(model_predictions[0])/len(model_predictions[0])} \n')
      
print(model_predictions[1])


