from aws import main as aws
from google_vision import main as google
import json
import os
import sys


folder='/home/mkelkar/tensor-flow/akhil/DRISHTI/images/meeting_bck'
for filename in os.listdir(folder):
        print folder+'/'+filename
        tags=""
        try:
            output= aws(folder+'/'+filename)
            print output
	    for i in output:
                if i['Confidence'] > 0.70:
                    tags=tags+i['Name']+" "

        except Exception as e:
            pass 
        print tags	

fo.close()
