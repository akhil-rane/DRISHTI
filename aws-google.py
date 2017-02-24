from aws import main as aws
from google_vision import main as google
import json
import os
import sys

filename=sys.argv[1]
output= aws(filename)
print output
print "AWS"
for i in output:
    if float(str(i['Confidence'])) > 95:
        print i['Name'] + " " + str(i['Confidence'])
print "Google"
output= google(filename)
list_google=output.split()
for word in list_google:
    print word+" "
#file.write(data)

