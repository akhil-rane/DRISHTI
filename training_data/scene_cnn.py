'''
author: akhil rane
'''
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import json
import argparse
import sys
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

learn = tf.contrib.learn

#CNN parameters set
FLAGS = None
MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0

#loading clarifai tags from google images
with open('meeting_data_bck', 'r') as myfile:
    meeting_data=myfile.read().replace('\n', '')
with open('cafeteria_data_bck', 'r') as myfile:
    cafeteria_data=myfile.read().replace('\n', '')
with open('park_data_bck', 'r') as myfile:
    park_data=myfile.read().replace('\n', '')
with open('parking_lot_data_bck', 'r') as myfile:
    parking_lot_data=myfile.read().replace('\n', '')

data=[meeting_data,cafeteria_data,park_data,parking_lot_data]
target=[0,1,2,3]

#Mr. CNN model starts here

def cnn_model(features, target):
  """2 layer ConvNet to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  target = tf.one_hot(target, 15, 1, 0)
  word_vectors = tf.contrib.layers.embed_sequence(
      features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
  word_vectors = tf.expand_dims(word_vectors, 3)
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.contrib.layers.convolution2d(
        word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
    # Add a RELU for non linearity.
    conv1 = tf.nn.relu(conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, POOLING_WINDOW, 1, 1],
        strides=[1, POOLING_STRIDE, 1, 1],
        padding='SAME')
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.contrib.layers.convolution2d(
        pool1, N_FILTERS, FILTER_SHAPE2, padding='VALID')
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  # Apply regular WX + B and classification.
  logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)

  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)





'''
#training the model using tf-idf and naive bayes classifier
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
feature_columns = skflow.infer_real_valued_columns_from_input(X_train_tfidf)
#clf = MultinomialNB().fit(X_train_tfidf, target)
#feature_columns = learn.infer_real_valued_columns_from_input(data)
clf = skflow.DNNClassifier(hidden_units=[10, 20, 10], n_classes=4,feature_columns=feature_columns)
clf.fit(X_train_counts, target,steps=10, batch_size=32)

#Scene prediction for new input image

input_image=sys.argv[1]
image_file = [input_image]
app = ClarifaiApp()
json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
json_obj=json.loads(json_data)
json_tags=json_obj['outputs'][0]['data']['concepts']
tags=''
for tag in json_tags:
	tags=tags+tag['name']+' '

tags=raw_input('Enter the tags:')
test_data=[tags]
X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
print predicted[0]
'''
#Use this code for model evaluation

#loading images taken by pranav in office

test_images_folder='/home/mkelkar/tensor-flow/akhil/DRISHTI/images'
test_data = []
test_target = []

app = ClarifaiApp()

for filename in os.listdir(test_images_folder+'/meeting'):
	image_file = [test_images_folder+'/meeting/'+filename]
	json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
	json_obj=json.loads(json_data)
	json_tags=json_obj['outputs'][0]['data']['concepts']
	tags=''
	for tag in json_tags:
		if tag['value'] > 0.90 :
			tags=tags+tag['name']+' '	
	test_data.append(tags)
	test_target.append(0)

for filename in os.listdir(test_images_folder+'/park'):
        image_file = [test_images_folder+'/park/'+filename]
        json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
        json_obj=json.loads(json_data)
        json_tags=json_obj['outputs'][0]['data']['concepts']
        tags=''
        for tag in json_tags:
		if tag['value'] > 0.90 :
                	tags=tags+tag['name']+' '
        test_data.append(tags)
        test_target.append(2)

for filename in os.listdir(test_images_folder+'/parking_lot'):
        image_file = [test_images_folder+'/parking_lot/'+filename]
        json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
        json_obj=json.loads(json_data)
        json_tags=json_obj['outputs'][0]['data']['concepts']
        tags=''
        for tag in json_tags:
		if tag['value'] > 0.90 :
                	tags=tags+tag['name']+' '
        test_data.append(tags)
        test_target.append(3)
'''
#Performance evaluation with test data
X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
print 'True Labels of test data:'
print test_target
print 'Predicted Labels of test data:'
print predicted
print 'Accuracy:'
print (accuracy_score(test_target, predicted))
'''

#model evaluation for cnn
x_train = data
y_train = target
x_test = test_data
y_test = test_target

# Process vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

# Build model
classifier = learn.Estimator(model_fn=cnn_model)

# Train and predict
classifier.fit(x_train, y_train, steps=300)
y_predicted = [
    p['class'] for p in classifier.predict(
        x_test, as_iterable=True)
]
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))
