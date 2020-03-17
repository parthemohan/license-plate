#Author:@Karthick Ramesh
from PIL import Image
import requests
import cv2
from io import BytesIO
import json
import pandas as pd

#Opening the JSON file and extracting the lines
with open('Indian_Number_plates.json', 'r') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]
i=0
xml_list=[]
#step 2:
#images are stored
for distro in data:
    url=distro["content"]
    im = Image.open(requests.get(url, stream=True).raw)
    im.save(str(i) + ".png")
i=0
xml_list=[]
#step:3
#Storing all the images and the co-ordinates into pandas data frame and convert it into csv file
for distro in data:
    i=i+1
    input=str(i)+".png"
    x1=distro["annotation"][0]['points'][0]["x"]
    y1=distro["annotation"][0]['points'][0]["y"]
    x2=distro["annotation"][0]['points'][1]["x"]
    y2=distro["annotation"][0]['points'][1]["y"]
    width=distro["annotation"][0]['imageWidth']
    height=distro["annotation"][0]['imageHeight']
    coords=(x1*width, y1*height, x2*width, y2*height)
    value = (input,width,height,"number_plate",int(x1*width),int( y1*height), int(x2*width), int(y2*height))
    xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv('train.csv', index=None)

