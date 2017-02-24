from aws import main as aws
from google_vision import main as google
import json
import os
import sys

foldername=sys.argv[1]
fo = open("data/%s_data" % foldername, "w")
folder='/home/mkelkar/tensor-flow/akhil/DRISHTI/images'
for filename in os.listdir('%s/%s' % (folder,foldername)):
	print folder+'/'+foldername+'/'+filename
        flag = 0
 	try:
            output= aws(folder+'/'+foldername+'/'+filename)
	    for i in output:
                if i['Confidence'] > 0.95:
                    fo.write(i['Name']+" ")     
                    print i['Name']+" "
                    flag = 1
        except Exception as e:
            pass
        try:
            output= google(folder+'/'+foldername+'/'+filename)
            list_google=output.split()
            for word in list_google:
                fo.write(word+" ")
                print word+" "
                flag = 2
            if flag != 0 :
                fo.write("\n")
                print ("\n")
        except Exception as e:
            pass 
        #file.write(data)	

fo.close()
