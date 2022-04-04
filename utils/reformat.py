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

    def load_data_folder(self, data_type):
        self.data_type_path = self.data_path.replace(".json", "_" + data_type + ".json")
        self.dials = {}

        if os.path.exists(self.data_type_path) and os.path.isfile(self.data_type_path):
            with open(self.data_type_path) as df:
                self.dials = json.loads(df.read().lower())
        else:
            data_dir = os.path.join(self.data_dir, data_type)
            sys.stdout.write('Extracting ' + data_type + ' data .... \n')
            for filename in tqdm(os.listdir(data_dir)):
                if 'dialog' in filename: # exclude schema.json
                    file_path = os.path.join(data_dir, filename)
                else:
                    continue
                if os.path.isfile(file_path):
                    data_json = open(file_path, 'r', encoding='utf-8')
                    data_in_file = json.loads(data_json.read().lower())
                    data_json.close()
                else:
                    continue

                for dial in data_in_file:
                    dial_id = data_type + "_" + dial["dialogue_id"]
                    if dial_id in self.dials:
                        pdb.set_trace()
                    self.dials[dial_id] = dial

            with open(self.data_type_path, "w") as df:
                json.dump(self.dials, df, indent=2)
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


class ReformatMultiWOZ21(ReformatBase):
    """docstring for ReformatMultiWOZ21"""
    def __init__(self, input_dir="./assets/"):
        super(ReformatMultiWOZ21, self).__init__()
        self.data_dir = os.path.join(input_dir, "multiwoz_dst/MULTIWOZ2.1/")
        self.reformat_data_dir = "./assets/mwoz21"
        self.slot_accm = True
        self.hist_accm = True


    def reformat(self):
        """
        following trade's code for normalizing multiwoz*
        now the data has format:
        file=[{
            "dialogue_idx": dial_id,
            "domains": [dom],
            "dialogue": [
                    {
                        "turn_idx": 0,
                        "domain": "hotel",
                        "system_transcript": "system response",
                        "transcript": "user utterance",
                        "system_act": [],
                        "belief_state": [{
                            "slots":[["domain-slot_type","slot_vale"]],
                            "act":  "inform"
                        }, ...], # accumulated
                        "turn_label": [["domain-slot_type","slot_vale"],...],    # for current turn
                    },
                    ...
                ],
                ...
            },
            ]
        and output with format like:
        file={
            dial_id:
                [
                    {
                        "dial_id": dial_id
                        "turn_num": 0,
                        "current_domain" : domain,
                        "potential_domains" : domain, domain2, ...   
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),
                        "context" : "<user> ... <system> ... <user> ..."
                    },...
                ]
            ...
            }
        """
        for mode in ["train", "dev", "test"]:
            self.data_trade_proc_path = os.path.join(self.data_dir, f"{mode}_dials_trade.json")
            self.load_json(data_path = self.data_trade_proc_path)

            self.dials_reformat = {}
            for dial in tqdm(self.dials):
                # self.dials_form[dial["dialogue_idx"]] = []
                context = []
                dial_id = dial["dialogue_idx"]
                dial_new = []

                for turn in dial["dialogue"]:                
                    turn_new = {
                                    "turn_num": turn["turn_idx"],   # turn number
                                    "dial_id" : dial_id,           # dial_id
                                    "current_domain" : turn["domain"],
                                    "potential_domains" : dial["domains"],
                                 }           
                    # # # slots/dialog states
                    slots_inf = []
                    if not self.slot_accm:
                        # # # dialog states only for the current turn, extracted based on "turn_label"
                        for slot in turn["turn_label"]:
                            domain    = slot[0].split("-")[0]
                            slot_type = slot[0].split("-")[1]
                            slot_val  = slot[1]
                            slots_inf += [domain, slot_type, slot_val, ","]
                    else:
                        # # # ACCUMULATED dialog states, extracted based on "belief_state"
                        for state in turn["belief_state"]:
                            if state["act"] == "inform":
                                domain = state["slots"][0][0].split("-")[0]
                                slot_type = state["slots"][0][0].split("-")[1]
                                slot_val  = state["slots"][0][1]
                                if "," in slot_val:
                                    if slot_val == "16,15":
                                        slot_val = "16:15"
                                slots_inf += [domain, slot_type, slot_val, ","]

                    turn_new["slots_inf"] = " ".join(slots_inf)

                    # # # dialog history
                    if turn["system_transcript"] != "":
                        context.append("<system> " + turn["system_transcript"])
                    
                    if not self.hist_accm:
                        context = context[-1:]

                    # # # adding current turn to dialog history
                    context.append("<user> " + turn["transcript"])
                    turn_new["context"] = " ".join(context)
                    dial_new.append(turn_new)

                self.dials_reformat[dial_id] = dial_new

            self.reformat_data_path = os.path.join(self.reformat_data_dir, f"{mode}.json")

            with open(self.reformat_data_path, "w") as tf:
                json.dump(self.dials_reformat, tf, indent=2)

class ReformatMultiWOZ22(ReformatBase):
    """docstring for ReformatMultiWOZ22"""
    def __init__(self, input_dir="./assets/"):
        super(ReformatMultiWOZ22, self).__init__()
        self.data_dir = os.path.join(input_dir, "multiwoz_dst/MULTIWOZ2.2/")
        self.data_path = os.path.join(self.data_dir, "data.json")
        self.reformat_data_dir = "./assets/mwoz"
        self.slot_accm = True

        self.val_path = os.path.join(self.data_dir, "valListFile.txt")
        self.test_path = os.path.join(self.data_dir, "testListFile.txt")
        self.val_list = self.load_txt(self.val_path)
        self.test_list = self.load_txt(self.test_path)
            

    def reformat(self):
        """
        file={
            dial_id: [
                    {
                        "turn_num": 0,
                        "utt": user utterance,
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),
                        "context" : "User: ... Sys: ..."
                    },
                    ...
                ],
                ...
            }
        """
        self.load_json()
        self.dials_form = {}
        self.dials_train, self.dials_dev, self.dials_test = {}, {}, {}
        
        for dial_id, dial in tqdm(self.dials.items()):
            context = []
            dial_new = []

            for turn_num in range(math.ceil(len(dial["log"]) / 2)):
                # # # turn number
                turn = {"turn_num": turn_num,
                        "dial_id" : dial_id}

                # # # user utterance
                user_utt = dial["log"][turn_num * 2]["text"]
                sys_resp = dial["log"][turn_num * 2 + 1]["text"]

                # # # dialog states, extracted based on "metadata", only in system side (turn_num * 2 + 1)
                slots_inf = []
                for domain, slot in dial["log"][turn_num * 2 + 1]["metadata"].items():
                    for slot_type, slot_val in slot["book"].items():
                        if slot_val != [] and slot_type != "booked":
                            slots_inf += [domain,
                                          slot_type,
                                          slot_val[0] + ","]
                    
                    for slot_type, slot_val in slot["semi"].items():
                        if slot_val != []:
                            slots_inf += [domain,
                                          slot_type,
                                          slot_val[0] + ","]

                
                turn["slots_inf"] = " ".join(slots_inf)
                turn["slots_err"] = ""
                # # adding current turn to dialog history
                context.append("<user> " + user_utt)
                # # # dialog history 
                turn["context"] = " ".join(context)
                # adding system response to next turn
                context.append("<system> " + sys_resp)

                dial_new.append(turn)

            if dial_id in self.test_list:
                self.dials_test[dial_id] = dial_new
            elif dial_id in self.val_list:
                self.dials_dev[dial_id] = dial_new
            else:
                self.dials_train[dial_id] = dial_new

        self.reformat_train_data_path = os.path.join(self.reformat_data_dir, "train.json")
        self.reformat_dev_data_path = os.path.join(self.reformat_data_dir, "dev.json")
        self.reformat_test_data_path = os.path.join(self.reformat_data_dir, "test.json")

        with open(self.reformat_train_data_path, "w") as tf:
            json.dump(self.dials_train, tf, indent=2)
        with open(self.reformat_dev_data_path, "w") as tf:
            json.dump(self.dials_dev, tf, indent=2)
        with open(self.reformat_test_data_path, "w") as tf:
            json.dump(self.dials_test, tf, indent=2)

class ReformatSGD(ReformatBase):
    """docstring for ReformatSGD"""
    def __init__(self, data_dir="./assets/google_sgd/"):
        super(ReformatSGD, self).__init__()
        # self.data_dir = "/checkpoint/kunqian/dstc8-schema-guided-dialogue/"
        self.data_dir =data_dir
        self.data_path = os.path.join(self.data_dir, "data.json")


    def reformat(self):
        """
        file={
            dial_id: [
                    {
                        "turn_num": 0,
                        "utt": user utterance,
                        "slots_inf": slot sequence ("dom slot_type1 slot_val1, dom slot_type2 ..."),
                        # "slots_err": slot sequence ("dom slot_type1 slot_type2, ..."),
                        "context" : "User: ... Sys: ..."
                    },
                    ...
                ],
                ...
            }
        """
        for data_type in ['train', 'dev', 'test']:
            self.load_data_folder(data_type)
            self.dials_form = {}
            sys.stdout.write('Reformating ' + data_type + ' data .... \n')
            for dial_id, dial in tqdm(self.dials.items()):
                self.dials_form[dial_id] = []
                turn_new = {}
                turn_num = 0
                bspan = {} # {dom:{slot_type:val, ...}, ...}
                context = []
                for turn in dial['turns']:
                    # turn number
                    turn_new['turn_num'] = turn_num
                    
                    if turn['speaker'] == 'system':
                        # turn_new['sys'] = self._tokenize_punc(turn['utterance'])
                        context.append("<system> " + self._tokenize_punc(turn['utterance']))

                    if turn['speaker'] == 'user':
                        context.append("<user> " + self._tokenize_punc(turn['utterance']))

                        # dialog history/context
                        turn_new["context"] = " ".join(context)

                        # belief span
                        turn_new['slots_inf'], bspan = \
                            self._extract_slots(bspan, turn['frames'], turn['utterance'], turn_new["context"])

                        # # user utterance
                        # turn_new['utt'] = self._tokenize_punc(turn['utterance'])
                    
                        self.dials_form[dial_id].append(turn_new)
                        turn_new = {}
                        turn_num += 1
                        
            # save reformatted dialogs
            self.reformat_data_path = os.path.join("./assets/sgd", data_type + ".json")
            with open(self.reformat_data_path, "w") as tf:
                json.dump(self.dials_form, tf, indent=2)


    def _extract_slots(self, bspan, frames, utt, context):
        """
        Input:
            bspan = {
                        dom:{
                            slot_type : slot_val, 
                            ...
                            }, 
                        ...
                    }

            frames = [
                        {
                            "service" : domain,
                            "slots"   : [
                                            {
                                                "exclusive_end" : idx,
                                                "slot" : slot_type,
                                                "start": idx
                                            }, 
                                            ...
                                        ],
                            "state"   : {"slot_values": {slot_type:[slot_val, ...], ...}}
                        },
                        ...
                     ]
        
        Notice that:
            frames["slots"] contains only non-categorical slots, while
            frames["state"] contains both non-categorical and categorical slots, 
            but it may contains multiple slot_vals for non-categorical slots.
            Therefore, we extract non-categorical slots based on frames["slots"]
            and extract categorical slots based on frames["state"]
        
        Output:
            formalize dialog states into string like:
                "restaurant area centre, restaurant pricerange cheap, ..."
        """

        dial_state = []
        for frame in frames:
            # extract Non-Categorical slots, based on frame["slots"]
            domain = frame["service"]
            if domain not in bspan:
                bspan[domain] = {}
            for slot in frame["slots"]:
                slot_type = slot["slot"]
                slot_val  = utt[slot["start"]: slot["exclusive_end"]] + ","
                bspan[domain][slot_type] = slot_val

            # extract Categorical slots, based on frame["state"]
            for slot_type in frame["state"]["slot_values"]:
                if slot_type not in bspan[domain]:
                    if len(set(frame["state"]["slot_values"][slot_type])) == 1:
                        bspan[domain][slot_type] = frame["state"]["slot_values"][slot_type][0] + ","
                    elif len(set(frame["state"]["slot_values"][slot_type])) > 1:
                        count = 0
                        slot_val_list = sorted(list(set(frame["state"]["slot_values"][slot_type])), key=lambda i: len(i))

                        for slot_val in slot_val_list:
                            # shown up in previous utt/bspan, referring to slots in other domain
                            if slot_val in utt:
                                bspan[domain][slot_type] = slot_val + ","
                                count += 1
                            # if count > 1:
                            #     print("utt contains non-categorical slot vals: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()

                        if count == 0:
                            for slot_val in slot_val_list:
                                # shown up in previous utt/bspan, referring to slots in other domain
                                if slot_val+"," in self._extract_all_slot_vals(bspan):
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1
                            # if count > 1:
                            #     print("multiple non-categorical slot vals: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()

                        if count == 0:
                            for slot_val in slot_val_list:
                                if slot_val in context and self._find_speaker(slot_val, context) == "user":
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1
                        
                            # if count > 1:
                            #     print("non-categorical slot vals in context user utt: ", slot_type)
                            #     print(frame["state"]["slot_values"][slot_type])
                            #     print(dial_id)
                            #     pdb.set_trace()
                        
                        if count == 0:
                            for slot_val in slot_val_list:
                                if slot_val in context:
                                    bspan[domain][slot_type] = slot_val + ","
                                    count += 1

                    # elif len(set(frame["state"]["slot_values"][slot_type])) == 0:
                    #     print("non-categorical slots with no value: ", slot_type)
                    #     print(frame["state"]["slot_values"][slot_type])
                    #     print(dial_id)
                    #     pdb.set_trace()

        # rewrite all the slots:
        for domain in bspan:
            for slot_type in bspan[domain]:
                dial_state += [domain, slot_type, bspan[domain][slot_type]]

        return " ".join(dial_state), bspan

    def _extract_all_slot_vals(self, bspan):
        """
        Input:
            bspan = {
                        dom:{
                            slot_type : slot_val, 
                            ...
                            }, 
                        ...
                    }
        To extract all the slot_val in this bspan.
        Return in a form of list
        """
        slot_val_list = []
        for dom in bspan:
            slot_val_list += list(bspan[dom].values())
        return slot_val_list

    def _find_speaker(self, slot_val, context):
        """
        assume the slot_val exists in the context, try
        to find out who is the first speaker mentioned 
        the slot_val
        context = "User: ... Sys: ... User: ... "
        """
        for utt in context.split("user:"):
            if slot_val in utt:
                if slot_val in utt.split("sys:")[0]:
                    return "user"
                else:
                    return "sys"
                    
    def _tokenize_punc(self, utt):
        """

        """
        corner_case = ['\.\.+', '!\.', '\$\.']
        for case in corner_case:
            utt = re.sub(case, ' .', utt)

        utt = re.sub('(?<! )\?+', ' ?', utt)
        utt = re.sub('(?<! )!+', ' !', utt)
        utt = re.sub('(?<! ),+(?= )', ' ,', utt)

        utt = re.sub('(?<=[a-z0-9])>(?=$)', ' .', utt)
        utt = re.sub('(\?>|>\?)', ' ?', utt)

        # utt = re.sub('(?<=[a-z0-9\]])(,|\.)$', ' .', utt)
        utt = re.sub('(?<!\.[a-z])(?<! )\.(?= )', ' .', utt)
        utt = re.sub('(?<!\.[a-z])(?<! )(,|\.)$', ' .', utt)
        utt = re.sub('(?<! )!$', ' !', utt)
        utt = re.sub('(?<! )\?$', ' ?', utt)

        utt = self.clean_text(utt)
        return utt

    def clean_text(self, text):
        text = text.strip()
        text = text.lower()
        text = text.replace(u"’", "'")
        text = text.replace(u"‘", "'")
        text = text.replace(';', ',')
        text = text.replace('"', ' ')
        text = text.replace('/', ' and ')
        text = text.replace("don't", "do n't")
        text = clean_time(text)
        baddata = { r'c\.b (\d), (\d) ([a-z])\.([a-z])': r'cb\1\2\3\4',
                            'c.b. 1 7 d.y': 'cb17dy',
                            'c.b.1 7 d.y': 'cb17dy',
                            'c.b 25, 9 a.q': 'cb259aq',
                            'isc.b 25, 9 a.q': 'is cb259aq',
                            'c.b2, 1 u.f': 'cb21uf',
                            'c.b 1,2 q.a':'cb12qa',
                            '0-122-336-5664': '01223365664',
                            'postcodecb21rs': 'postcode cb21rs',
                            r'i\.d': 'id',
                            ' i d ': 'id',
                            'Telephone:01223358966': 'Telephone: 01223358966',
                            'depature': 'departure',
                            'depearting': 'departing',
                            '-type': ' type',
                            r"b[\s]?&[\s]?b": "bed and breakfast",
                            "b and b": "bed and breakfast",
                            r"guesthouse[s]?": "guest house",
                            r"swimmingpool[s]?": "swimming pool",
                            "wo n\'t": "will not",
                            " \'d ": " would ",
                            " \'m ": " am ",
                            " \'re' ": " are ",
                            " \'ll' ": " will ",
                            " \'ve ": " have ",
                            r'^\'': '',
                            r'\'$': '',
                                    }
        for tmpl, good in baddata.items():
            text = re.sub(tmpl, good, text)

        text = re.sub(r'([a-zT]+)\.([a-z])', r'\1 . \2', text)   # 'abc.xyz' -> 'abc . xyz'
        text = re.sub(r'(\w+)\.\.? ', r'\1 . ', text)   # if 'abc. ' -> 'abc . '
        return text
        

class ReformatGSIM(object):
    """docstring for ReformatGSIM"""
    def __init__(self, arg):
        super(ReformatGSIM, self).__init__()
        self.arg = arg


class ReformatABCD(object):
    """docstring for ReformatABCD"""
    def __init__(self, arg):
        super(ReformatABCD, self).__init__()
        self.arg = arg
        

class ReformatTaskMaster(object):
    """docstring for ReformatTaskMaster"""
    def __init__(self, arg):
        super (ReformatTaskMaster, self).__init__()
        self.arg = arg
        

class ReformatReddit(object):
    """docstring for ReformatReddit"""
    def __init__(self, arg):
        super(ReformatReddit, self).__init__()
        self.arg = arg
        

def main():
    reformatmwoz21 = ReformatMultiWOZ21()
    reformatmwoz21.reformat()

if __name__ == "__main__":
    main()
