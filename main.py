import json
import pandas as pd

"""
longIntentList = list()
longPatternList = list()

# Load the data
with open('data_full.json', 'r') as longFile, \
    open('longFormatted.json', 'w') as longTarget:
    data = json.load(longFile)

    for intent, content in data.items():
        for x in range(len(content)):
            longPatternList.append(content[x][0])
            longIntentList.append(content[x][1])

    noDupIntents = list(set(longIntentList))
    print(noDupIntents)
"""

X = pd.read_csv('train.csv')

patterns = list(X['text'])
intents = list(X['category'])
noDupIntents = sorted(list(set(intents)))

jsonDict = {
    "intents": []
}
for intent in range(len(noDupIntents)):
    addDict = {
        "tag": str(noDupIntents[intent]),
        "patterns": [pattern.rstrip("\n") for n, pattern in enumerate(patterns) if intents[n] == noDupIntents[intent]],
        "responses": ["add responses here"]
    }
    jsonDict["intents"].append(addDict)

# Serializing json
json_object = json.dumps(jsonDict, indent=4)

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)

print(noDupIntents[:11])
print(noDupIntents[12:22])
print(noDupIntents[23:33])
print(noDupIntents[34:44])
print(noDupIntents[45:55])
print(noDupIntents[56:66])
print(noDupIntents[67:77])
