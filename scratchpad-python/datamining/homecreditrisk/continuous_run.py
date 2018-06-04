import os
import sys
import json
import time

try:
    rel_path = sys.argv[1]
except IndexError as e:
    rel_path = None

if rel_path is None:
    path = os.path.join(os.getcwd(), 'setup')
else:
    path = os.path.join(os.getcwd(), rel_path)

job_list = []

while True:
    for dir in os.listdir(path):
        setup_json_path = os.path.join(path, dir, 'setup.json')
        if os.path.isfile(setup_json_path):
            with open(setup_json_path, 'r') as setup_json_file:
                setup_json = json.load(setup_json_file)

            if 'epochs' in setup_json and \
                'expected_epochs' in setup_json['others'] and \
                setup_json['epochs'] < setup_json['others']['expected_epochs']:

                os.system('python train_model.py \"%s\" 5' % setup_json_path)

    time.sleep(3)
