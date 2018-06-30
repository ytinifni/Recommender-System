import json
from pprint import pprint
#This is the library for handling big JSON files:
import ijson
import pandas as pd

jsonFile = 'm1.json'

#this is not working for now
def load_json(filename):
    with open(filename, 'r') as fd:
        parser = ijson.parse(fd)
        ret = {'builders': {}}
        for prefix, event, value in parser:
            print(event)
            if (prefix, event) == ('builders', 'map_key'):
                buildername = value
                ret['builders'][buildername] = {}
            elif prefix.endswith('.shortname'):
                ret['builders'][buildername]['shortname'] = value

        return ret

if __name__ == "__main__":
    res = load_json('m2.json')
