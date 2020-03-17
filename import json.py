import json
import pandas as pd
import Number_plate_detection as n

with open('Indian_Number_plates.json') as f:
    data = f.readlines()

    data = [json.loads(line)for line in data] #convert string to dict format
    #df = pd.read_json(data) # Load into dataframe
    i=0
    for distro in data:
        i=i+1
        print(distro["annotation"])

print(i)
target=n.target
target.show()