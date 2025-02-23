
"""

################################################################################
# TODO: Fill in your codes                                                     #
# Import packages or modules                                                   #
################################################################################
import tensorflow as tf
import numpy as np
from tensorflow import keras

"""## Part I: Explore Word Embedding (15%)

Word embeddings are useful representation of words that capture information about word meaning as well as location. They are used as a fundamental component for downstream NLP tasks, e.g., text classification. In this part, we will explore the embeddings produced by [GloVe (global vectors for word representation)](https://nlp.stanford.edu/projects/glove/). It is simlar to Word2Vec but differs in their underlying methodology: in GloVe, word embeddings are learned based on global word-word co-occurrence statistics. Both Word2Vec and GloVe tend to produce vector-space embeddings that perform similarly in downstream NLP tasks.

We first load the GloVe vectors
"""

import gensim.downloader as api
# download the model and return as object ready for use
glove_word_vectors = api.load('glove-wiki-gigaword-100')

"""Take a look at the vocabulary size and dimensionality of the embedding space"""

print('vocabulary size = ', len(glove_word_vectors.index_to_key))
print('embedding dimensionality = ', glove_word_vectors['happy'].shape)

"""
What is embedding exactly?"""

# Check word embedding for 'happy'
# You can access the embedding of a word with glove_word_vectors[word] if word
# is in the vocabulary
glove_word_vectors['happy']

"""With word embeddings learned from GloVe or Word2Vec, words with similar semantic meanings tend to have vectors that are close together. Please code and calculate the **cosine similarities** between words based on their embeddings (i.e., word vectors).

For each of the following words in occupation, compute its cosine similarty to 'woman' and its similarity to 'man' and check which gender is more similar.

*occupation = {homemaker, nurse, receptionist, librarian, socialite, hairdresser, nanny, bookkeeper, stylist, housekeeper, maestro, skipper, protege, philosopher, captain, architect, financier, warrior, broadcaster, magician}*



"""

################################################################################
# TODO: Fill in your codes                                                     #                                                          #
################################################################################
from sklearn.metrics.pairwise import cosine_similarity
woman_vector = glove_word_vectors['woman'].reshape(1, -1)
man_vector = glove_word_vectors['man'].reshape(1, -1)

occupation = ['homemaker', 'nurse', 'receptionist', 'librarian', 'socialite', 'hairdresser', 'nanny',
              'bookkeeper', 'stylist', 'housekeeper', 'maestro', 'skipper', 'protege', 'philosopher',
              'captain', 'architect', 'financier', 'warrior', 'broadcaster', 'magician']

similarities = {'occupation': [], 'woman': [], 'man': []}

for job in occupation:
    job_vector = glove_word_vectors[job].reshape(1, -1)
    woman_similarity = cosine_similarity(job_vector, woman_vector)[0][0]
    man_similarity = cosine_similarity(job_vector, man_vector)[0][0]

    similarities['occupation'].append(job)
    similarities['woman'].append(round(woman_similarity, 2))
    similarities['man'].append(round(man_similarity, 2))

print('{:<15} {:<10} {:<10}'.format('Occupation', 'Similarity to Woman', 'Similarity to Man'))
for i in range(len(occupation)):
    print('{:<15} {:<10} {:<10}'.format(similarities['occupation'][i], similarities['woman'][i], similarities['man'][i]))

"""## Part II Understand contextual word embedding using BERT (15%)

A big difference between Word2Vec and BERT is that Word2Vec learns context-free word representations, i.e., the embedding for 'orange' is the same in "I love eating oranges" and in "The sky turned orange". BERT, on the other hand, produces contextual word presentations, i.e., embeddings for the same word in different contexts are different.

For example, let us compare the context-based embedding vectors for 'orange' in the following three sentences using Bert:
* "I love eating oranges"
* "My favorite fruits are oranges and apples"
* "The sky turned orange"

Same as in "Lab 5 BERT", we use the BERT model and tokenizer from the Huggingface transformer library ([1](https://huggingface.co/course/chapter1/1), [2](https://huggingface.co/docs/transformers/quicktour))
"""

# Note that we need to install the latest version of transformers
# Due to problems we encountered in class and reported here
# https://discuss.huggingface.co/t/pretrain-model-not-accepting-optimizer/76209
# https://github.com/huggingface/transformers/issues/29470

!pip install --upgrade transformers
import transformers
print(transformers.__version__)

from transformers import BertTokenizer, TFBertModel

"""We use the 'bert-base-cased' from Huggingface as the underlying BERT model and the associated tokenizer."""

bert_model = TFBertModel.from_pretrained('bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_sentences = ["I love eating oranges",
                     "My favorite fruits are oranges and apples",
                     "The sky turned orange"]

"""Let us start by tokenizing the example sentences."""

# Check how Bert tokenize each sentence
# This helps us identify the location of 'orange' in the tokenized vector
for sen in example_sentences:
  print(bert_tokenizer.tokenize(sen))

"""Notice that the prefix '##' indicates that the token is a continuation of the previous one. This also helps us identify location of 'orange' in the tokenized vector, e.g., 'orange' is the 4th token in the first sentence. Note that here the tokenize() function just splits a text into words, and doesn't add a 'CLS' (classification token) or a 'SEP' (separation token) to the text.

Next, we use the tokenizer to transfer the example sentences to input that the Bert model expects.
"""

bert_inputs = bert_tokenizer(example_sentences,
                             padding=True,
                             return_tensors='tf')

bert_inputs

"""So there are actually three outputs: the input ids (starting with '101' for the '[CLS]' token), the token_type_ids which are usefull when one has distinct segments, and the attention masks which are used to mask out padding tokens.

Please refer to our Lab 4 for more details about input_ids, token_type_ids, and attention_masks.

More resources:
*    https://huggingface.co/docs/transformers/preprocessing
*    https://huggingface.co/docs/transformers/tokenizer_summary

Now, let us get the BERT encoding of our example sentences.
"""

bert_outputs = bert_model(bert_inputs)

print('shape of first output: \t\t', bert_outputs[0].shape)
print('shape of second output: \t', bert_outputs[1].shape)

"""There are two outputs here: one with dimensions [3, 10, 768] and one with [3, 768]. The first one [batch_size, sequence_length, embedding_size] is the output of the last layer of the Bert model and are the contextual embeddings of the words in the input sequence. The second output [batch_size, embedding_size] is the embedding of the first token of the sequence (i.e., classification token).

Note you can also get the first output through bert_output.last_hidden_state (see below, also check https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.TFBertModel)

We need the first output to get contextualized embeddings for 'orange' in each sentence.
"""

bert_outputs[0]

bert_outputs.last_hidden_state

"""Now, we get the embeddings of 'orange' in each sentence by simply finding the 'orange'-token positions in the embedding output and extract the proper components:"""

orange_1 = bert_outputs[0][0, 4]
orange_2 = bert_outputs[0][1, 5]
orange_3 = bert_outputs[0][2, 4]

oranges = [orange_1, orange_2, orange_3]

"""We calculate pair-wise cosine similarities:"""

def cosine_similarities(vecs):
    for v_1 in vecs:
        similarities = ''
        for v_2 in vecs:
            similarities += ('\t' + str(np.dot(v_1, v_2)/
                np.sqrt(np.dot(v_1, v_1) * np.dot(v_2, v_2)))[:4])
        print(similarities)

cosine_similarities(oranges)

"""The similarity metrics make sense. The 'orange' in "The sky turned orange" is different from the rest.

Next, please compare the contextual embedding vectors of 'bank' in the following four sentences:


*   "I need to bring my money to the bank today"
*   "I will need to bring my money to the bank tomorrow"
*   "I had to bank into a turn"
*   "The bank teller was very nice"


**Inline Question #1:**

- Please calculate the pair-wise cosine similarities between 'bank' in the four sentences and fill in the table below. (Note, bank_i represent bank in the i_th sentence)
- Please explain the results. Does it make sense?

**Your Answer:**

| `similarity`|  bank_1  |  bank_2  |  bank_3  |  bank_4  |
|-------------|----------|----------|----------|----------|
| bank_1      |    1.0      | 0.99         | 0.59        |  0.86        |
| bank_2      |    0.99      | 1.0         | 0.59        |  0.87        |
| bank_3      |    0.59      | 0.59         | 1.0         |  0.62        |
| bank_4      |    0.86     |  0.87        |  0.62        |   0.99       |

bank_1 and bank_2: These sentences are quite similar in meaning, just differing in the timing. Hence, it's expected to see a very high similarity between their contextual embeddings.

bank_1 and bank_3: These sentences are semantically quite different. In the first sentence, 'bank' is used in the sense of a financial institution, whereas in the third sentence, 'bank' is used as a verb meaning to tilt or incline. Thus, the cosine similarity is relatively lower, indicating less similarity in meaning.

bank_1 and bank_4: Although these sentences are different, they both relate to financial institutions. However, the contextual usage differs slightly. Hence, we observe a relatively high cosine similarity, but slightly lower compared to bank_1 and bank_2.

bank_2 and bank_3: Similar to bank_1 and bank_3, these sentences have different semantic meanings, resulting in lower cosine similarity.

bank_2 and bank_4: Similar to bank_1 and bank_4, these sentences relate to financial institutions, but with slightly different contextual usage. Hence, the cosine similarity is relatively high.

bank_3 and bank_4: These sentences are quite different in meaning, with 'bank' used as a verb in one and as a noun in the other. Thus, the cosine similarity is relatively low.
"""

bert_model = TFBertModel.from_pretrained('bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_sentences = [
    "I need to bring my money to the bank today",
    "I will need to bring my money to the bank tomorrow",
    "I had to bank into a turn",
    "The bank teller was very nice"
]

bert_inputs = bert_tokenizer(example_sentences,
                             padding=True,
                             return_tensors='tf')

bert_outputs = bert_model(bert_inputs)

bank_1 = bert_outputs[0][0, 9]
bank_2 = bert_outputs[0][1, 10]
bank_3 = bert_outputs[0][2, 4]
bank_4 = bert_outputs[0][3,2]

banks = [bank_1, bank_2, bank_3, bank_4]

def cosine_similarities(vecs):
    for v_1 in vecs:
        similarities = ''
        for v_2 in vecs:
            similarities += ('\t' + str(np.dot(v_1, v_2) /
                             (np.linalg.norm(v_1) * np.linalg.norm(v_2)))[:4])
        print(similarities)

# Calculate and print cosine similarities
cosine_similarities(banks)

"""## Part III Text classification

In this part, you will build text classifiers that try to infer whether tweets from [@realDonaldTrump](https://twitter.com/realDonaldTrump) were written by Trump himself or by a staff person.
This is an example of binary classification on a text dataset.

It is known that Donald Trump uses an Android phone, and it has been observed that some of his tweets come from Android while others come from other devices (most commonly iPhone). It is widely believed that Android tweets are written by Trump himself, while iPhone tweets are written by other staff. For more information, you can read this [blog post by David Robinson](http://varianceexplained.org/r/trump-tweets/), written prior to the 2016 election, which finds a number of differences in the style and timing of tweets published under these two devices. (Some tweets are written from other devices, but for simplicity the dataset for this assignment is restricted to these two.)

This is a classification task known as "authorship attribution", which is the task of inferring the author of a document when the authorship is unknown. We will see how accurately this can be done with linear classifiers using word features.

You might find it familiar: Yes! We are using the same data set as your homework 2 from MSBC 5180.

### Tasks

In this section, you will build two text classifiers: one with a traditional machine learning method that you studied in MSBC.5190 and one with a deep learning method.


*   For the first classifier, you can use any non-deep learning based methods. You can use your solution to Homework 2 of MSBC 5180 here.
*   For the second classifier, you may try the following methods
    *    Fine-tune BERT (similar to our Lab 5 Fine-tune BERT for Sentiment Analysis)
    *    Use pre-trained word embedding (useful to check: https://keras.io/examples/nlp/pretrained_word_embeddings/)
    *    Train a deep neural network (e.g., CNN, RNN, Bi-LSTM) from scratch, similar to notebooks from our textbook:
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/dense_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/convolutional_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/rnn_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/lstm_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/bi_lstm_sentiment_classifier.ipynb
    *   There are also lots of useful resources on Keras website: https://keras.io/examples/nlp/

You may want to split the current training data to train and validation to help model selection. Please do not use test data for model selection.

### Load the Data Set
"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir ("/content/drive/My Drive/Modern Ai homework 2")

"""#### Sample code to load raw text###

Please download `tweets.train.tsv` and `tweets.test.tsv` from Canvas (Module Assignment) and upload them to Google Colab. Here we load raw text data to text_train and text_test.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#training set
df_train = pd.read_csv('tweets.train.tsv', sep='\t', header=None)

text_train = df_train.iloc[0:, 1].values.tolist()
Y_train = df_train.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_train = np.array([1 if v == 'Android' else 0 for v in Y_train])

df_test = pd.read_csv('tweets.test.tsv', sep='\t', header=None)
text_test = df_test.iloc[0:, 1].values.tolist()
Y_test = df_test.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_test = np.array([1 if v == 'Android' else 0 for v in Y_test])

"""Let us take a quick look of some training examples"""

text_train[:5]

y_train[:5]

"""#### Sample code to preprocess data for BERT (only needed if you decide to fine-tune BERT) ####

The pre-processing step is similar to Lab 5.

Feel free to dispose it if you want to preprocess the data differently and use methods other than BERT.
"""

# The longest text in the data is 75 and we use it as the max_length
max_length = 75
x_train = bert_tokenizer(text_train,
              max_length=75,
              truncation=True,
              padding='max_length',
              return_tensors='tf')

y_train = np.array([1 if v == 'Android' else 0 for v in Y_train])

x_test = bert_tokenizer(text_test,
              max_length=75,
              truncation=True,
              padding='max_length',
              return_tensors='tf')

y_test = np.array([1 if v == 'Android' else 0 for v in Y_test])

"""### Your Solution 1: A traditional machine learning approach (30%)

Features Used: The features used in this model are the counts of each word in the vocabulary extracted from the text data using the CountVectorizer.

Model Performance on Test Data: The accuracy on the test set can be obtained from the accuracy score calculated using the true labels and the predicted labels. This accuracy score gives us an idea of how well the model generalizes to unseen data.
"""

################################################################################
# TODO: Fill in your codes                                                     #
################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
vectorizer = CountVectorizer(max_features=5000)
X_train_bow = vectorizer.fit_transform(text_train)
X_test_bow = vectorizer.transform(text_test)
model = LogisticRegression()

model.fit(X_train_bow, y_train)

y_pred = model.predict(X_test_bow)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy on test set:", accuracy)

################################################################################
# TODO: Fill in your codes                                                     #
################################################################################
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)

sequences_train = tokenizer.texts_to_sequences(text_train)
sequences_test = tokenizer.texts_to_sequences(text_test)

x_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
x_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_length))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=32)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)
