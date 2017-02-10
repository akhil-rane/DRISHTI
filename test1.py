from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import json

app = ClarifaiApp()

image_files = ["/root/test_drishti/images/meeting/meeting_77.jpg"]
json_data= json.dumps(app.tag_files(image_files,model='aaa03c23b3724a16a56b629203edc62c'))
json_obj=json.loads(json_data)
print (json_obj['outputs'][0]['data']['concepts'])
#app.tag_urls(['https://static1.squarespace.com/static/5495db14e4b08142c223fd25/t/54b1818ae4b0f72291b3355c/1420919295341/osoyoos-spirit-ridge-meeting-04.jpg'])
