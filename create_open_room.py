import urllib.request
import json
import datetime

url = "http://localhost:8000/create_room"
past_time = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")

data = {
    "text": "SHARE\nYOUR\nHEART",
    "columns": 5,
    "open_time": past_time,
    "host_name": "AI_HELPER"
}

req = urllib.request.Request(url, 
                             data=json.dumps(data).encode('utf-8'), 
                             headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
아