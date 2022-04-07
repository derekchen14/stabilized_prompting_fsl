#!/usr/bin/env python3
#
import os, sys, json
import math, argparse, random, re
from tqdm import tqdm
import pdb

class ReformatBase(object):
    """docstring for ReformatBase"""
    def __init__(self):
        super(ReformatBase, self).__init__()
        
    def load_json(self, data_path=None):
        if data_path is None:
            data_path = self.data_path
        with open(data_path) as df:
            self.dials = json.loads(df.read().lower())
    
    def load_txt(self, file_path):
        with open(file_path) as df:
            data=df.read().lower().split("\n")
            data.remove('')
        return data

    def load_data_folder(self, data_type, data_dir):
        dials = []
        data_type_dir = os.path.join(self.data_dir, data_type)
        sys.stdout.write('Extracting ' + data_type + ' data .... \n')
        for filename in tqdm(os.listdir(data_type_dir)):
            if 'dialog' in filename: # exclude schema.json
                file_path = os.path.join(data_type_dir, filename)
            else:
                continue
            if os.path.isfile(file_path):
                data_json = open(file_path, 'r', encoding='utf-8')
                data_in_file = json.loads(data_json.read().lower())
                data_json.close()
            else:
                continue

            dials += data_in_file

        return dials

    def save_dials(self):
        with open(self.reformat_data_path, "w") as tf:
            json.dump(self.dials_form, tf, indent=2)

    def reformat(self):
        """ Default reformatting does nothing """
        raise NotImplementedError

class ReformatABCD(ReformatBase):
    def __init__(self, input_dir="./assets/"):
        super().__init__()
        self.filename = 'abcd_v1.1.json'
        self.splits = ['train', 'dev', 'test']
    
    def reformat(self):
        file_path = os.path.join(self.input_dir, self.filename)
        all_data = json.load(open(file_path, 'r'))

        for split in self.splits:
            data = all_data[split]
            trimmed = []

            for convo in data:
                turns = []
                for delex, orig in zip(convo['delexed'], convo['original']):
                    speaker, text = orig
                    assert(speaker == delex['speaker'])
                    tc = delex['turn_count']
                    targets = delex['targets']

                    new_turn = {'speaker': speaker, 'text': text, 'turn_count': tc, 'targets': targets}
                    turns.append(new_turn)

                new_convo = {'convo_id': convo['convo_id'], 'conversation': turns, 'scene': convo['scenario']}
                trimmed.append(new_convo)

            save_path = os.path.join(self.input_dir, f"{split}.json")
        json.dump(trimmed, open(save_path, 'w'))
        print(f"completed {split}")


class ReformatMultiWOZ22(ReformatBase):
    """docstring for ReformatMultiWOZ22"""
    def __init__(self, input_dir="./assets/"):
        super(ReformatMultiWOZ22, self).__init__()
        self.data_dir = os.path.join(input_dir, "multiwoz_dst/MULTIWOZ2.2/")
        self.reformat_data_dir = "./assets/mwoz"


    def reformat(self):
        for data_type in ['train', 'dev', 'test']:
            dials = self.load_data_folder(data_type, self.data_dir)
            reformat_data_path = os.path.join(self.reformat_data_dir, data_type + ".json")
            with open(reformat_data_path, "w") as tf:
                json.dump(dials, tf, indent=2)


class ReformatSGD(ReformatMultiWOZ22):
    """docstring for ReformatSGD"""
    def __init__(self, input_dir="./assets/"):
        super(ReformatSGD, self).__init__()
        self.data_dir = os.path.join(input_dir, "multiwoz_dst/google_sgd/")
        self.reformat_data_dir = "./assets/sgd"



class ReformatGSIM(object):
    """docstring for ReformatGSIM"""
    def __init__(self, arg):
        super(ReformatGSIM, self).__init__()
        self.arg = arg

class ReformatTaskMaster(object):
    """docstring for ReformatTaskMaster"""
    def __init__(self, arg):
        super (ReformatTaskMaster, self).__init__()
        self.arg = arg
        

class ReformatDSTC(object):
    """docstring for ReformatDSTC"""
    def __init__(self, arg):
        super(ReformatDSTC, self).__init__()
        self.arg = arg
        

def main():
    reformatmwoz21 = ReformatMultiWOZ21()
    reformatmwoz21.reformat()

if __name__ == "__main__":
    main()
