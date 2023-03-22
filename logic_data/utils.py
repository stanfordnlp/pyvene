from collections import OrderedDict
import random
import pickle
import pandas as pd
import re
from tqdm import tqdm
import copy
import numpy as np
import torch

FALSE_TOKEN_ID = 0
TRUE_TOKEN_ID = 1
INPUT_PREFIX_TOKEN_ID = 2
OUTPUT_PREFIX_TOKEN_ID = 3
SEPARATOR_TOKEN_ID = 4
PADDING_TOKEN_ID = 5
BOS_TOKEN_ID = 6
EOS_TOKEN_ID = 7

class CFG(OrderedDict):
    def __init__(self, *args):
        super().__init__(map(lambda s: s.replace(' ', '').split('->'), args))
        
    def __repr__(self):
        return '\n'.join('{} -> {}'.format(k, v) for k, v in self.items())

    def getProductions(self, symbol):
        return self[symbol].split('|')

# Depth-first walk through tree, selecting random productions
def generateSentence(cfg, start='S'):
    string = []
    def dfs(root):
        local_str = ''
        prod = random.choice(cfg.getProductions(root))
        for char in prod:
            if char in cfg:
                result = dfs(char)
                if result:
                    string.append(result)
            else:
                local_str += char
        return local_str

    dfs(start)
    return ' '.join(string)

# Example CFG found online
L = [
    'S -> CLAUSES',
    'CLAUSES -> CLAUSE CONJ CLAUSE',
    'CLAUSE -> LPR VAR EQ VAR RPR',
    'CONJ -> "and" | "or"',
    'EQ -> "==" | "!="',
    'VAR -> "a" | "b" | "c"',
    'LPR -> "("',
    'RPR -> ")"',
]

# Replacing variable names for simpler parsing
table = OrderedDict([
    ('CLAUSES', 'A'),
    ('CLAUSE',  'B'),
    ('CONJ',    'C'),
    ('EQ',      'D'),
    ('VAR',     'E'),
    ('LPR',     'F'),
    ('RPR',     'G')
])

conj_re = re.compile(r"""
    ^
    \s*
    \(
    \s*(\w+?)\s*(?:==|!=)\s*(\w+?)\s*
    \)
    \s*$""", re.VERBOSE)



def sample_var(exclude, low=10, high=50257):
    sampled_var = random.randint(low, high-1)
    while sampled_var in exclude:
        sampled_var = random.randint(low, high-1)
    return sampled_var

def parse(clauses):
    conjs = re.split(r"\s*(?:and|or)\s*", clauses)
    data = []
    for conj in conjs:
        if conj_re.search(conj):
            LVAR, RVAR = conj_re.search(conj).groups()
            EQ = "==" if "==" in conj else "!="
            d = {
                "L" : LVAR,
                "R" : RVAR,
                "EQ" : EQ
            }
            data += [d]
    return data


def sample_constituent_values(
    clauses,
    final_value=None,
):
    if final_value == None:
        final_value = random.choice([True, False])

    if "and" in clauses:
        data = parse(clauses)
        if final_value == True:
            data[0]["VAL"] = True
            data[1]["VAL"] = data[0]["VAL"]
        else:
            data[0]["VAL"] = True if random.random() >= 0.5 else False
            data[1]["VAL"] = not data[0]["VAL"]
    elif "or" in clauses:
        data = parse(clauses)
        if final_value:
            data[0]["VAL"] = True if random.random() >= 0.5 else False
            data[1]["VAL"] = random.choice([True, False]) if data[0]["VAL"] else True
        else:
            data[0]["VAL"] = False
            data[1]["VAL"] = data[0]["VAL"]
    else:
        data = parse(clauses)
        data[0]["VAL"] = final_value
        
    return data[0]["VAL"], data[1]["VAL"]


def sample_demonstrations_for_clauses_forward(
    clauses,
    n,
    partial_value_assignment=None
):

    value_assignments = []
    for _ in range(n):
        value_assignment = {}
        
        if partial_value_assignment is not None:
            for var, val in partial_value_assignment.items():
                value_assignment[var] = val
            unassigned = {'a', 'b', 'c'} - set(list(value_assignment.keys()))
            while len(unassigned) > 0:
                # assigning values.
                unassigned = list(unassigned)
                assigning_var = random.choice(unassigned)
                assigned = list(value_assignment.keys())
                assigned_val = list(value_assignment.values())
                matching_var = random.choice(assigned)
                _equal = True if random.random() < 0.5 else False
                assigning_val = value_assignment[matching_var] if _equal else sample_var(exclude=assigned_val)
                value_assignment[assigning_var] = assigning_val
                unassigned = {'a', 'b', 'c'} - set(list(value_assignment.keys()))
        else:
            rotary_index = random.choice([1,2,3])
            if rotary_index == 1:
                #abc
                ab_equal = True if random.random() < 0.5 else False
                bc_equal = True if random.random() < 0.5 else False
                a = sample_var(exclude=[])
                b = a if ab_equal else sample_var(exclude=[a])
                c = b if bc_equal else sample_var(exclude=[b])
            elif rotary_index == 2:
                #bca
                bc_equal = True if random.random() < 0.5 else False
                ca_equal = True if random.random() < 0.5 else False
                b = sample_var(exclude=[])
                c = b if bc_equal else sample_var(exclude=[b])
                a = c if ca_equal else sample_var(exclude=[c])
            elif rotary_index == 3:
                #cab
                ca_equal = True if random.random() < 0.5 else False
                ab_equal = True if random.random() < 0.5 else False
                c = sample_var(exclude=[])
                a = c if ca_equal else sample_var(exclude=[c])
                b = a if ab_equal else sample_var(exclude=[a])  

            value_assignment['a'] = a
            value_assignment['b'] = b
            value_assignment['c'] = c
        
        conjs = re.split(r"\s*(?:and|or)\s*", clauses)
        left_args = re.split(r"\s*(?:!=|==)\s*", conjs[0])
        right_args = re.split(r"\s*(?:!=|==)\s*", conjs[1])
        if "!=" in conjs[0]:
            value_assignment["LEFT_EQ"] = "!="
            LEFT_VAL = value_assignment[left_args[0].strip(" (")] != value_assignment[left_args[1].strip(" )")]
        elif "==" in conjs[0]:
            value_assignment["LEFT_EQ"] = "=="
            LEFT_VAL = value_assignment[left_args[0].strip(" (")] == value_assignment[left_args[1].strip(" )")]
        if "!=" in conjs[1]:
            value_assignment["RIGHT_EQ"] = "!="
            RIGHT_VAL = value_assignment[right_args[0].strip(" (")] != value_assignment[right_args[1].strip(" )")]
        elif "==" in conjs[1]:
            value_assignment["RIGHT_EQ"] = "=="
            RIGHT_VAL = value_assignment[right_args[0].strip(" (")] == value_assignment[right_args[1].strip(" )")]
        if "and" in clauses:
            value_assignment["LOGIC"] = "and"
            output = LEFT_VAL and RIGHT_VAL
        elif "or" in clauses:
            value_assignment["LOGIC"] = "or"
            output = LEFT_VAL or RIGHT_VAL
        value_assignment["LEFT_VAL"] = LEFT_VAL
        value_assignment["RIGHT_VAL"] = RIGHT_VAL
        value_assignment['output'] = output
        value_assignment['clause'] = clauses

        value_assignments += [value_assignment]
        # print(value_assignment)
    return value_assignments


def sample_demonstration_for_clauses(
    clauses,
    final_value_in=None,
    partial_value_assignment=None,
):

    success = False
    
    while not success:
        try:
            if final_value_in is None:
                final_value = random.choice([True, False])
            else:
                final_value = final_value_in
            if "and" in clauses:
                data = parse(clauses)
                if final_value == True:
                    data[0]["VAL"] = True
                    data[1]["VAL"] = data[0]["VAL"]
                else:
                    data[0]["VAL"] = True if random.random() >= 0.5 else False
                    data[1]["VAL"] = not data[0]["VAL"]
            elif "or" in clauses:
                data = parse(clauses)
                if final_value:
                    data[0]["VAL"] = True if random.random() >= 0.5 else False
                    data[1]["VAL"] = random.choice([True, False]) if data[0]["VAL"] else True
                else:
                    data[0]["VAL"] = False
                    data[1]["VAL"] = data[0]["VAL"]
            else:
                data = parse(clauses)
                data[0]["VAL"] = final_value

            used_var = set([])
            if partial_value_assignment is None:
                value_assignment = {}
            else:
                value_assignment = copy.deepcopy(partial_value_assignment)
                for k, v in partial_value_assignment.items():
                    used_var.add(v)
            for d in data:
                if (d["EQ"] == "==" and d["VAL"] == True) or \
                        (d["EQ"] == "!=" and d["VAL"] == False):
                    if d['L'] in value_assignment and d['R'] in value_assignment:
                        pass
                    elif d['L'] in value_assignment:
                        value_assignment[d['R']] = value_assignment[d['L']]
                    elif d['R'] in value_assignment:
                        value_assignment[d['L']] = value_assignment[d['R']]
                    else:
                        value_assignment[d['L']] = value_assignment[d['L']] if d['L'] in value_assignment else sample_var(used_var)
                        value_assignment[d['R']] = value_assignment[d['L']]
                        used_var.add(value_assignment[d['L']])
                elif (d["EQ"] == "==" and d["VAL"] == False) or \
                        (d["EQ"] == "!=" and d["VAL"] == True):
                    if d['L'] in value_assignment and d['R'] in value_assignment:
                        pass
                    elif d['L'] in value_assignment:
                        value_assignment[d['R']] = sample_var(used_var)
                        assert value_assignment[d['R']] != value_assignment[d['L']]
                        used_var.add(value_assignment[d['R']])
                    elif d['R'] in value_assignment:
                        value_assignment[d['L']] = sample_var(used_var)
                        assert value_assignment[d['L']] != value_assignment[d['R']]
                        used_var.add(value_assignment[d['L']])
                    else:
                        value_assignment[d['L']] = sample_var(used_var)
                        used_var.add(value_assignment[d['L']])
                        value_assignment[d['R']] = sample_var(used_var)
                        used_var.add(value_assignment[d['R']])

            for d in data:
                if (d["EQ"] == "==" and d["VAL"] == True) or \
                        (d["EQ"] == "!=" and d["VAL"] == False):
                    assert value_assignment[d['L']] == value_assignment[d['R']]
                elif (d["EQ"] == "==" and d["VAL"] == False) or \
                        (d["EQ"] == "!=" and d["VAL"] == True):
                    assert value_assignment[d['L']] != value_assignment[d['R']]

            if "and" in clauses:
                value_assignment["LOGIC"] = "and"
                assert final_value == (data[0]["VAL"] and data[1]["VAL"])
            elif "or" in clauses:
                value_assignment["LOGIC"] = "or"
                assert final_value == (data[0]["VAL"] or data[1]["VAL"])
                
            success = True
        except:
            pass # resample!
        
        value_assignment["LEFT_VAL"] = data[0]["VAL"]
        value_assignment["RIGHT_VAL"] = data[1]["VAL"]
        value_assignment["LEFT_EQ"] = data[0]["EQ"]
        value_assignment["RIGHT_EQ"] = data[1]["EQ"]
    value_assignment['output'] = final_value
    value_assignment['clause'] = clauses
    
    assert _eval(clauses, value_assignment) == final_value
    
    return value_assignment
    # we need to assert check

def sample_demonstrations_for_clauses(
    clauses,
    final_values,
    partial_value_assignment=None,
):
    demos = []
    for i in range(len(final_values)):
        demo = sample_demonstration_for_clauses(
            clauses, final_values[i], partial_value_assignment=partial_value_assignment
        )
        demos += [demo]
    return demos

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def _eval(
    clauses,
    assignments
):
    eval_clauses = clauses.replace('a ', str(assignments['a']))
    eval_clauses = eval_clauses.replace('b ', str(assignments['b']))
    eval_clauses= eval_clauses.replace('c ', str(assignments['c']))
    return eval(eval_clauses)

def sample_factual_demonstrations(clauses_pool, n_samples, n_examples, tokenizer):
    all_input_ids = []
    all_output_ids = []
    all_clauses = []

    for i in tqdm(range(n_samples)):
        clauses = random.choice(clauses_pool)
        demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples
        )
        # listify
        input_ids = [tokenizer.bos_token_id]
        output_ids = [tokenizer.bos_token_id]
        for d in demostrations:
            output = tokenizer.false_token_id if d['output'] == False else tokenizer.true_token_id
            input_ids += [tokenizer.input_token_id, d['a'], d['b'], d['c'], tokenizer.output_token_id, output, tokenizer.sep_token_id]
            output_ids += [-100, -100, -100, -100, -100, output, -100]
            assert len(input_ids) == len(output_ids)
        input_ids += [tokenizer.eos_token_id]
        output_ids += [tokenizer.eos_token_id]
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]
        all_clauses += [clauses]
    
    return all_input_ids, all_output_ids, all_clauses

def function_aligment_sampler(
    clauses, n_training_examples, n_examples = 7, shared_train = True, source_clauses=None
):
    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_input_ids = []
    all_source_output_ids = []
    all_source_clauses = []

    all_ctf_output_ids = [] # this one does not have input ids, etc..
        
    for i in tqdm(range(n_training_examples)):
        base_train_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
        source_train_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            n_examples-1
        )
        
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        source_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        ctf_val = _eval(
            source_clauses,
            { 
                'a': base_test_demostrations[0]['a'],
                'b': base_test_demostrations[0]['b'],
                'c': base_test_demostrations[0]['c'],
            }
        )

        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_input_ids = [BOS_TOKEN_ID]
        source_output_ids = [BOS_TOKEN_ID]
        for d in source_train_demostrations+source_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_output_ids)
        source_input_ids += [EOS_TOKEN_ID]
        source_output_ids += [EOS_TOKEN_ID]
        all_source_input_ids += [source_input_ids]
        all_source_output_ids += [source_output_ids]
        all_source_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]

    return all_base_input_ids, all_base_output_ids, all_base_clauses, \
        all_source_input_ids, all_source_output_ids, all_source_clauses, \
        all_ctf_output_ids


def whole_aligment_sampler(
    clauses, n_training_examples, n_examples = 7, shared_train = True, source_clauses=None
):
    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_input_ids = []
    all_source_output_ids = []
    all_source_clauses = []

    all_ctf_output_ids = [] # this one does not have input ids, etc..
        
    for i in tqdm(range(n_training_examples)):
        base_train_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
        source_train_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            n_examples-1
        )
        
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        source_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        ctf_val = source_test_demostrations[0]['output']

        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_input_ids = [BOS_TOKEN_ID]
        source_output_ids = [BOS_TOKEN_ID]
        for d in source_train_demostrations+source_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_output_ids)
        source_input_ids += [EOS_TOKEN_ID]
        source_output_ids += [EOS_TOKEN_ID]
        all_source_input_ids += [source_input_ids]
        all_source_output_ids += [source_output_ids]
        all_source_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]

    return all_base_input_ids, all_base_output_ids, all_base_clauses, \
        all_source_input_ids, all_source_output_ids, all_source_clauses, \
        all_ctf_output_ids

def left_aligment_sampler(
    clauses, n_training_examples, n_examples = 7, shared_train = True, source_clauses=None
):
    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_input_ids = []
    all_source_output_ids = []
    all_source_clauses = []

    all_ctf_output_ids = [] # this one does not have input ids, etc..
    
    if shared_train:
        shared_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
    if source_clauses is None:
        source_clauses = clauses
    else:
        assert shared_train == False
        
    for i in tqdm(range(n_training_examples)):
        if shared_train:
            base_train_demostrations = shared_demostrations
            
            source_train_demostrations = shared_demostrations
        else:
            base_train_demostrations = sample_demonstrations_for_clauses_forward(
                clauses,
                n_examples-1
            )
            source_train_demostrations = sample_demonstrations_for_clauses_forward(
                source_clauses,
                n_examples-1
            )
        
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        source_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        if "and" in clauses:
            ctf_val = source_test_demostrations[0]['LEFT_VAL'] and base_test_demostrations[0]['RIGHT_VAL']
        else:
            ctf_val = source_test_demostrations[0]['LEFT_VAL'] or base_test_demostrations[0]['RIGHT_VAL']
        
        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_input_ids = [BOS_TOKEN_ID]
        source_output_ids = [BOS_TOKEN_ID]
        for d in source_train_demostrations+source_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_output_ids)
        source_input_ids += [EOS_TOKEN_ID]
        source_output_ids += [EOS_TOKEN_ID]
        all_source_input_ids += [source_input_ids]
        all_source_output_ids += [source_output_ids]
        all_source_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]

    return {"base_input_ids" : all_base_input_ids,
            "base_output_ids" : all_base_output_ids,
            "source_left_input_ids" : all_source_input_ids,
            "source_left_output_ids" : all_source_output_ids,
            "counterfacut_output_ids": all_ctf_output_ids,
            "clauses" : all_base_clauses,
            "intervention_ids": [0 for i in range(len(all_base_input_ids))]}

def right_aligment_sampler(
    clauses, n_training_examples, 
    n_examples = 7, shared_train = True, source_clauses=None
):
    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_input_ids = []
    all_source_output_ids = []
    all_source_clauses = []

    all_ctf_output_ids = [] # this one does not have input ids, etc..
    
    if shared_train:
        shared_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
    if source_clauses is None:
        source_clauses = clauses
    else:
        assert shared_train == False
        
    for i in tqdm(range(n_training_examples)):
        if shared_train:
            base_train_demostrations = shared_demostrations
            
            source_train_demostrations = shared_demostrations
        else:
            base_train_demostrations = sample_demonstrations_for_clauses_forward(
                clauses,
                n_examples-1
            )
            source_train_demostrations = sample_demonstrations_for_clauses_forward(
                source_clauses,
                n_examples-1
            )
        
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        source_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        if "and" in clauses:
            ctf_val = source_test_demostrations[0]['RIGHT_VAL'] and base_test_demostrations[0]['LEFT_VAL']
        else:
            ctf_val = source_test_demostrations[0]['RIGHT_VAL'] or base_test_demostrations[0]['LEFT_VAL']
        
        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_input_ids = [BOS_TOKEN_ID]
        source_output_ids = [BOS_TOKEN_ID]
        for d in source_train_demostrations+source_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_output_ids)
        source_input_ids += [EOS_TOKEN_ID]
        source_output_ids += [EOS_TOKEN_ID]
        all_source_input_ids += [source_input_ids]
        all_source_output_ids += [source_output_ids]
        all_source_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]

    return {"base_input_ids" : all_base_input_ids,
            "base_output_ids" : all_base_output_ids,
            "source_right_input_ids" : all_source_input_ids,
            "source_right_output_ids" : all_source_output_ids,
            "counterfacut_output_ids": all_ctf_output_ids,
            "clauses" : all_base_clauses,
            "intervention_ids": [1 for i in range(len(all_base_input_ids))]}

def left_right_aligment_sampler(
    clauses, n_training_examples, 
    n_examples = 7, shared_train = True, source_clauses=None
):
    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_left_input_ids = []
    all_source_left_output_ids = []
    all_source_left_clauses = []

    all_source_right_input_ids = []
    all_source_right_output_ids = []
    all_source_right_clauses = []
    
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    
    if shared_train:
        shared_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
    if source_clauses is None:
        source_clauses = clauses
    else:
        assert shared_train == False
        
    for i in tqdm(range(n_training_examples)):
        if shared_train:
            base_train_demostrations = shared_demostrations
            
            source_left_train_demostrations = shared_demostrations
            source_right_train_demostrations = shared_demostrations
        else:
            base_train_demostrations = sample_demonstrations_for_clauses_forward(
                clauses,
                n_examples-1
            )
            source_left_train_demostrations = sample_demonstrations_for_clauses_forward(
                source_clauses,
                n_examples-1
            )
            source_right_train_demostrations = sample_demonstrations_for_clauses_forward(
                source_clauses,
                n_examples-1
            )
            
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        source_left_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        source_right_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1
        )
        if "and" in clauses:
            ctf_val = source_left_test_demostrations[0]['LEFT_VAL'] and source_right_test_demostrations[0]['RIGHT_VAL']
        else:
            ctf_val = source_left_test_demostrations[0]['LEFT_VAL'] or source_right_test_demostrations[0]['RIGHT_VAL']
        
        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_left_input_ids = [BOS_TOKEN_ID]
        source_left_output_ids = [BOS_TOKEN_ID]
        for d in source_left_train_demostrations+source_left_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_left_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_left_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_left_output_ids)
        source_left_input_ids += [EOS_TOKEN_ID]
        source_left_output_ids += [EOS_TOKEN_ID]
        all_source_left_input_ids += [source_left_input_ids]
        all_source_left_output_ids += [source_left_output_ids]
        all_source_left_clauses += [clauses]
        
        source_right_input_ids = [BOS_TOKEN_ID]
        source_right_output_ids = [BOS_TOKEN_ID]
        for d in source_right_train_demostrations+source_right_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_right_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_right_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_right_output_ids)
        source_right_input_ids += [EOS_TOKEN_ID]
        source_right_output_ids += [EOS_TOKEN_ID]
        all_source_right_input_ids += [source_right_input_ids]
        all_source_right_output_ids += [source_right_output_ids]
        all_source_right_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]

    return {"base_input_ids" : all_base_input_ids,
            "base_output_ids" : all_base_output_ids,
            "source_left_input_ids" : all_source_left_input_ids,
            "source_left_output_ids" : all_source_left_output_ids,
            "source_right_input_ids" : all_source_right_input_ids,
            "source_right_output_ids" : all_source_right_output_ids,
            "counterfacut_output_ids": all_ctf_output_ids,
            "clauses" : all_base_clauses,
            "intervention_ids": [2 for i in range(len(all_base_input_ids))]}

def left_identity_alignment_sampler(
    clauses, n_training_examples, n_examples = 7, shared_train = True, source_clauses=None
):

    all_base_input_ids = []
    all_base_output_ids = []
    all_base_clauses = []

    all_source_input_ids = []
    all_source_output_ids = []
    all_source_clauses = []

    all_ctf_output_ids = [] # this one does not have input ids, etc..
    
    diff_probe = 0
    
    if shared_train:
        shared_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            n_examples-1
        )
    if source_clauses is None:
        source_clauses = clauses
    else:
        assert shared_train == False
        
    for i in tqdm(range(n_training_examples)):
        if shared_train:
            base_train_demostrations = shared_demostrations
            
            source_train_demostrations = shared_demostrations
        else:
            base_train_demostrations = sample_demonstrations_for_clauses_forward(
                clauses,
                n_examples-1
            )
            source_train_demostrations = sample_demonstrations_for_clauses_forward(
                source_clauses,
                n_examples-1
            )
        
        base_test_demostrations = sample_demonstrations_for_clauses_forward(
            clauses,
            1
        )
        left_clause = base_test_demostrations[0]['clause'].split(f" {base_test_demostrations[0]['LOGIC']} ")[0]
        left_first_arg = left_clause.strip("() ").split(f" {base_test_demostrations[0]['LEFT_EQ']} ")[0].strip()
        left_second_arg = left_clause.strip("() ").split(f" {base_test_demostrations[0]['LEFT_EQ']} ")[-1].strip()
        base_left_first_var = base_test_demostrations[0][left_first_arg]
        base_left_second_var = base_test_demostrations[0][left_second_arg]
        
        right_clause = base_test_demostrations[0]['clause'].split(f" {base_test_demostrations[0]['LOGIC']} ")[1]
        right_first_arg = right_clause.strip("() ").split(f" {base_test_demostrations[0]['RIGHT_EQ']} ")[0].strip()
        right_second_arg = right_clause.strip("() ").split(f" {base_test_demostrations[0]['RIGHT_EQ']} ")[-1].strip()
        base_right_first_var = base_test_demostrations[0][right_first_arg]
        base_right_second_var = base_test_demostrations[0][right_second_arg]
        
        fixed_args = list({'a', 'b', 'c'} - {left_first_arg})
        value_assignment = {}
        for arg in fixed_args:
            value_assignment[arg] = base_test_demostrations[0][arg]
        ctf_value_assignment = sample_demonstrations_for_clauses_forward(
            clauses,
            1,
            partial_value_assignment=value_assignment,
        )[0]
        ctf_val = ctf_value_assignment['output']
        ctf_arg_val = ctf_value_assignment[left_first_arg]
        source_value_assignment = {
            left_first_arg: ctf_arg_val,
        }
        source_test_demostrations = sample_demonstrations_for_clauses_forward(
            source_clauses,
            1,
            partial_value_assignment=source_value_assignment,
        )
        
        # listify
        base_input_ids = [BOS_TOKEN_ID]
        base_output_ids = [BOS_TOKEN_ID]
        for d in base_train_demostrations+base_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            base_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            base_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(base_input_ids) == len(base_output_ids)
        base_input_ids += [EOS_TOKEN_ID]
        base_output_ids += [EOS_TOKEN_ID]
        all_base_input_ids += [base_input_ids]
        all_base_output_ids += [base_output_ids]
        all_base_clauses += [clauses]   

        # listify
        source_input_ids = [BOS_TOKEN_ID]
        source_output_ids = [BOS_TOKEN_ID]
        for d in source_train_demostrations+source_test_demostrations:
            output = FALSE_TOKEN_ID if d['output'] == False else TRUE_TOKEN_ID
            source_input_ids += [INPUT_PREFIX_TOKEN_ID, d['a'], d['b'], d['c'], OUTPUT_PREFIX_TOKEN_ID, output, SEPARATOR_TOKEN_ID]
            source_output_ids += [-100, -100, -100, -100, -100, output, -100] # no label to predict!
            assert len(source_input_ids) == len(source_output_ids)
        source_input_ids += [EOS_TOKEN_ID]
        source_output_ids += [EOS_TOKEN_ID]
        all_source_input_ids += [source_input_ids]
        all_source_output_ids += [source_output_ids]
        all_source_clauses += [clauses]
        
        # counterfactuals, we ONLY need one single label.
        ctf_output_ids = [-100]
        for d in base_train_demostrations:
            ctf_output_ids += [-100, -100, -100, -100, -100, -100, -100] # no label to predict!
        ctf_output = FALSE_TOKEN_ID if ctf_val == False else TRUE_TOKEN_ID
        ctf_output_ids += [-100, -100, -100, -100, -100, ctf_output, -100]
        ctf_output_ids += [-100]
        all_ctf_output_ids += [ctf_output_ids]
        
        assert _eval(
            clauses, 
            ctf_value_assignment
        ) == ctf_val
        
    return all_base_input_ids, all_base_output_ids, all_base_clauses, \
        all_source_input_ids, all_source_output_ids, all_source_clauses, \
        all_ctf_output_ids