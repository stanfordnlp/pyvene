import random
import copy
import inspect
import itertools
import torch
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


class CausalModel:
    def __init__(
        self,
        variables,
        values,
        parents,
        functions,
        timesteps=None,
        equiv_classes=None,
        pos={},
    ):
        self.variables = variables
        self.variables.sort()
        self.values = values
        self.parents = parents
        self.children = {var: [] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)
        self.functions = functions
        self.start_variables = []
        self.timesteps = timesteps
        for variable in self.variables:
            assert variable in self.values
            assert variable in self.children
            assert variable in self.functions
            assert len(inspect.getfullargspec(self.functions[variable])[0]) == len(
                self.parents[variable]
            )
            if timesteps is not None:
                assert variable in timesteps
            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    assert variable in self.children[variable2]
                    if timesteps is not None:
                        assert timesteps[variable2] < timesteps[variable]
                if variable2 in self.children[variable]:
                    assert variable in parents[variable2]
                    if timesteps is not None:
                        assert timesteps[variable2] > timesteps[variable]
            if len(self.parents) == 0:
                self.start_variables.append(variable)

        self.inputs = [var for var in self.variables if len(parents[var]) == 0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)
        if self.timesteps is not None:
            self.timesteps = timesteps
        else:
            self.timesteps, self.end_time = self.generate_timesteps()
            for output in self.outputs:
                self.timesteps[output] = self.end_time
        self.variables.sort(key=lambda x: self.timesteps[x])
        self.run_forward()
        self.pos = pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.pos == None:
            self.pos = dict()
        for var in self.variables:
            if var not in pos:
                pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        if equiv_classes is not None:
            self.equiv_classes = equiv_classes
        else:
            self.equiv_classes = {}
        for var in self.variables:
            if var in self.inputs or var in self.equiv_classes:
                continue
            self.equiv_classes[var] = {val: [] for val in self.values[var]}
            for parent_values in itertools.product(
                *[self.values[par] for par in self.parents[var]]
            ):
                value = self.functions[var](*parent_values)
                self.equiv_classes[var][value].append(
                    {par: parent_values[i] for i, par in enumerate(self.parents[var])}
                )

    def generate_timesteps(self):
        timesteps = {input: 0 for input in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(timesteps)
            for parent in timesteps:
                if timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            timesteps = copytimesteps
            step += 1
        for var in self.variables:
            assert var in timesteps
        # return all timesteps and timestep of root
        return timesteps, step - 2

    def marginalize(self, target):
        pass

    def print_structure(self, pos=None):
        G = nx.DiGraph()
        G.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, with_labels=True, node_color="green", pos=self.pos)
        plt.show()

    def find_live_paths(self, intervention):
        actual_setting = self.run_forward(intervention)
        paths = {1: [[variable] for variable in self.variables]}
        step = 2
        while True:
            paths[step] = []
            for path in paths[step - 1]:
                for child in self.children[path[-1]]:
                    actual_cause = False
                    for value in self.values[path[-1]]:
                        newintervention = copy.deepcopy(intervention)
                        newintervention[path[-1]] = value
                        counterfactual_setting = self.run_forward(newintervention)
                        if counterfactual_setting[child] != actual_setting[child]:
                            actual_cause = True
                    if actual_cause:
                        paths[step].append(copy.deepcopy(path) + [child])
            if len(paths[step]) == 0:
                break
            step += 1
        del paths[1]
        return paths

    def print_setting(self, total_setting, display=None):
        labeler = lambda var: var + ": " + str(total_setting[var]) \
            if display is None or display[var] \
            else var
        relabeler = {
            var: labeler(var) for var in self.variables
        }
        G = nx.DiGraph()
        G.add_edges_from(
            [
                (parent, child)
                for child in self.variables
                for parent in self.parents[child]
            ]
        )
        plt.figure(figsize=(10, 10))
        G = nx.relabel_nodes(G, relabeler)
        newpos = dict()
        if self.pos is not None:
            for var in self.pos:
                newpos[relabeler[var]] = self.pos[var]
        nx.draw_networkx(G, with_labels=True, node_color="green", pos=newpos)
        plt.show()

    def run_forward(self, intervention=None):
        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        step = 0
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if intervention is not None and variable in intervention:
                    total_setting[variable] = intervention[variable]
                else:
                    total_setting[variable] = self.functions[variable](
                        *[total_setting[parent] for parent in self.parents[variable]]
                    )
            length = len(list(total_setting.keys()))
        return total_setting

    def run_interchange(self, input, source_interventions):
        interchange_intervention = copy.deepcopy(input)
        for var in source_interventions:
            setting = self.run_forward(source_interventions[var])
            interchange_intervention[var] = setting[var]
        return self.run_forward(interchange_intervention)

    def add_variable(
        self, variable, values, parents, children, function, timestep=None
    ):
        if timestep is not None:
            assert self.timesteps is not None
            self.timesteps[variable] = timestep
        for parent in parents:
            assert parent in self.variables
        for child in children:
            assert child in self.variables
        self.parents[variable] = parents
        self.children[variable] = children
        self.values[variable] = values
        self.functions[variable] = function

    def sample_intervention(self, mandatory=None):
        intervention = {}
        while len(intervention.keys()) == 0:
            for var in self.variables:
                if var in self.inputs or var in self.outputs:
                    continue
                if random.choice([0, 1]) == 0:
                    intervention[var] = random.choice(self.values[var])
        return intervention

    def sample_input(self, mandatory=None):
        input = {var: random.sample(self.values[var], 1)[0] for var in self.inputs}
        total = self.run_forward(intervention=input)
        while mandatory is not None and not mandatory(total):
            input = {var: random.sample(self.values[var], 1)[0] for var in self.inputs}
            total = self.run_forward(intervention=input)
        return input

    def sample_input_tree_balanced(self, output_var=None, output_var_value=None):
        assert output_var is not None or len(self.outputs) == 1
        if output_var is None:
            output_var = self.outputs[0]
        if output_var_value is None:
            output_var_value = random.choice(self.values[output_var])

        def create_input(var, value, input={}):
            parent_values = random.choice(self.equiv_classes[var][value])
            for parent in parent_values:
                if parent in self.inputs:
                    input[parent] = parent_values[parent]
                else:
                    create_input(parent, parent_values[parent], input)
            return input

        input_setting = create_input(output_var, output_var_value)
        for input_var in self.inputs:
            if input_var not in input_setting:
                input_setting[input_var] = random.choice(self.values[input_var])
        return input_setting

    def get_path_maxlen_filter(self, lengths):
        def check_path(total_setting):
            input = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input)
            m = max([l for l in paths.keys() if len(paths[l]) != 0])
            if m in lengths:
                return True
            return False

        return check_path

    def get_partial_filter(self, partial_setting):
        def compare(total_setting):
            for var in partial_setting:
                if total_setting[var] != partial_setting[var]:
                    return False
            return True

        return compare

    def get_specific_path_filter(self, start, end):
        def check_path(total_setting):
            input = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input)
            for k in paths:
                for path in paths[k]:
                    if path[0] == start and path[-1] == end:
                        return True
            return False

        return check_path

    def input_to_tensor(self, setting):
        result = []
        for input in self.inputs:
            temp = torch.tensor(setting[input]).float()
            if len(temp.size()) == 0:
                temp = torch.reshape(temp, (1,))
            result.append(temp)
        return torch.cat(result)

    def output_to_tensor(self, setting):
        result = []
        for output in self.outputs:
            temp = torch.tensor(float(setting[output]))
            if len(temp.size()) == 0:
                temp = torch.reshape(temp, (1,))
            result.append(temp)
        return torch.cat(result)

    def generate_factual_dataset(
        self,
        size,
        sampler=None,
        filter=None,
        device="cpu",
        input_function=None,
        output_function=None,
        return_tensors=True,
    ):
        if sampler is None:
            sampler = self.sample_input
        
        if input_function is None:
            input_function = self.input_to_tensor
        if output_function is None:
            output_function = self.output_to_tensor

        examples = []
        while len(examples) < size:
            example = dict()
            input = sampler()
            if filter is None or filter(input):
                output = self.run_forward(input)
                if return_tensors:
                    example['input_ids'] = input_function(input).to(device)
                    example['labels'] = output_function(output).to(device)
                else:
                    example['input_ids'] = input
                    example['labels'] = output
                examples.append(example)

        return examples

    def generate_counterfactual_dataset(
        self,
        size,
        intervention_id,
        batch_size,
        sampler=None,
        intervention_sampler=None,
        filter=None,
        device="cpu",
        input_function=None,
        output_function=None,
        return_tensors=True,
    ):
        if input_function is None:
            input_function = self.input_to_tensor
        if output_function is None:
            output_function = self.output_to_tensor

        maxlength = len(
            [
                var
                for var in self.variables
                if var not in self.inputs and var not in self.outputs
            ]
        )
        if sampler is None:
            sampler = self.sample_input
        if intervention_sampler is None:
            intervention_sampler = self.sample_intervention
        examples = []
        while len(examples) < size:
            intervention = intervention_sampler()
            if filter is None or filter(intervention):
                for _ in range(batch_size):
                    example = dict()
                    base = sampler()
                    sources = []
                    source_dic = {}
                    for var in self.variables:
                        if var not in intervention:
                            continue
                        # sample input to match sampled intervention value
                        source = sampler(output_var=var, output_var_value=intervention[var])
                        if return_tensors:
                            sources.append(self.input_to_tensor(source))
                        else:
                            sources.append(source)
                        source_dic[var] = source
                    for _ in range(maxlength - len(sources)):
                        if return_tensors:
                            sources.append(torch.zeros(self.input_to_tensor(base).shape))
                        else:
                            sources.append({})

                    if return_tensors:
                        example["labels"] = self.output_to_tensor(
                            self.run_interchange(base, source_dic)
                        ).to(device)
                        example["base_labels"] = self.output_to_tensor(
                            self.run_forward(base)
                        ).to(device)
                        example["input_ids"] = self.input_to_tensor(base).to(device)
                        example["source_input_ids"] = torch.stack(sources).to(device)
                        example["intervention_id"] = torch.tensor(
                            [intervention_id(intervention)]
                        ).to(device)
                    else:
                        example['labels'] = self.run_interchange(base, source_dic)
                        example['base_labels'] = self.run_forward(base)
                        example['input_ids'] = base
                        example['source_input_ids'] = sources
                        example['intervention_id'] = [intervention_id(intervention)]

                    examples.append(example)
        return examples


def simple_example():
    variables = ["A", "B", "C"]
    values = {variable: [True, False] for variable in variables}
    parents = {"A": [], "B": [], "C": ["A", "B"]}

    def A():
        return True

    def B():
        return False

    def C(a, b):
        return a and b

    functions = {"A": A, "B": B, "C": C}
    model = CausalModel(variables, values, parents, functions)
    model.print_structure()
    print("No intervention:\n", model.run_forward(), "\n")
    model.print_setting(model.run_forward())
    print(
        "Intervention setting A and B to TRUE:\n",
        model.run_forward({"A": True, "B": True}),
    )
    print("Timesteps:", model.timesteps)


if __name__ == "__main__":
    simple_example()
