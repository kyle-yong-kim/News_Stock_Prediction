import numpy as np
import pandas as pd
import json

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def mainLoop():
    fileLoc = "C:\\Temp\\newsData.json"
    
    newsJson = loadJson(fileLoc)

    # try to make lambda function to do this job
    count_dec, count_inc = 0, 0
    count_dec = sum(x["label"] == 0 for x in newsJson)
    count_inc = sum(x["label"] == 1 for x in newsJson)

    # x = lambda a, b : a * b
    # count_dec = len(list(filter(lambda x: x["label"] == 0, newsJson)))
    # count_inc = len(list(filter(lambda x: x["label"] == 1, newsJson)))

mainLoop()

print("done")