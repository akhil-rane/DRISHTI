'''
author: akhil rane
'''
import numpy
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
import tensorflow.contrib.learn as skflow

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
target=[0,1,2,4]

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
'''
input_image=sys.argv[1]
image_file = [input_image]
app = ClarifaiApp()
json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
json_obj=json.loads(json_data)
json_tags=json_obj['outputs'][0]['data']['concepts']
tags=''
for tag in json_tags:
	tags=tags+tag['name']+' '
'''
tags=raw_input('Enter the tags:')
test_data=[tags]
X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
print predicted[0]

#Use this code for model evaluation

#loading images taken by pranav in office
'''
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
	test_target.append('meeting')

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
        test_target.append('park')

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
        test_target.append('parking_lot')

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
