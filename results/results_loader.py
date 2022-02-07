import os
import csv
import json
import re


class ResultsReader:
    #list_float_types = {'Sparseness', 'SensitivityN'}
    list_list_float_types = {'Monotonicity Nguyen', 'SensitivityN', 'Pixel-Flipping', 'Local Lipschitz Estimate',
                             'Faithfulness Estimate', 'Faithfulness Correlation', 'Avg-Sensitivity', 'Random Logit',
                             'Complexity', 'Max-Sensitivity', 'Sparseness'}
    list_list_int_types = {'EffectiveComplexity', 'Nonsensitivity'}
    list_list_bool = {'Completeness', 'Monotonicity Arya'}
    list_dict_int_list_float_types = {'Region Perturbation', 'Selectivity'}
    list_dict_string_list_float_types = {'Model Parameter Randomisation'}
    list_dict_int_dict_int_list_float_types = {'Continuity Test'}
    
    def read(self, file, metric_name):
        if metric_name in self.list_list_float_types or metric_name in self.list_list_int_types:
            return self.read_list_list_float_types(file)
        elif metric_name in self.list_list_bool:
            return self.read_list_list_bool(file)
        elif metric_name in self.list_dict_int_list_float_types:
            return self.read_list_dict_int_list_float_types(file)
        elif metric_name in self.list_dict_string_list_float_types:
            return self.read_list_dict_string_list_float_types(file)
        elif metric_name in self.list_dict_int_dict_int_list_float_types:
            return self.read_list_dict_int_dict_int_list_float_types(file)
        else:
            raise Exception(f'{metric_name} is not supported !')
    
    def read_list_list_float_types(self, file):
        csv_reader = csv.reader(file)
        results = []
        for row in csv_reader:
            results.append([float(v) for v in row])
        return results
    
    def read_list_list_bool(self, file):
        csv_reader = csv.reader(file)
        results = []
        for row in csv_reader:
            results.append([v == 'True' for v in row])
        return results
    
    def read_list_dict_int_list_float_types(self, file):
        results = []
        for line in file:
            corrected_line = re.sub(r'(\d+):', r'"\1":', line[1:-2])
            results.append(json.loads(corrected_line))
        return results
    
    def read_list_dict_string_list_float_types(self, file):
        results = []
        for line in file:
            corrected_line = line[1:-2].replace('\'', '\"')
            results.append(json.loads(corrected_line))
        return results
    
    def read_list_dict_int_dict_int_list_float_types(self, file):
        results = []
        for line in file:
            corrected_line = re.sub(r'(\d+):', r'"\1":', line[1:-2])
            results.append(json.loads(corrected_line))
        return results
    

class ResutlsTransformer:
    list_list_types = {'Monotonicity Nguyen', 'SensitivityN', 'Local Lipschitz Estimate',
                       'Faithfulness Estimate', 'Faithfulness Correlation', 'Avg-Sensitivity', 'Random Logit',
                       'Complexity', 'Max-Sensitivity', 'Sparseness', 'EffectiveComplexity', 'Nonsensitivity',
                       'Completeness', 'Monotonicity Arya'}
    # need to investigate before applying a extended transformation, only flatten to corresponding number of inputs
    list_dict_types = {'Region Perturbation', 'Selectivity', 'Continuity Test'}
    # need to investigate before applying a transformation
    no_transformation_types = {'Pixel-Flipping', 'Model Parameter Randomisation'}
    
    def transform(self, results, metric_name):
        if metric_name in self.list_list_types:
            return self.transform_list_list_types(results)
        elif metric_name in self.list_dict_types:
            return self.transform_list_dict_types(results)
        elif metric_name in self.no_transformation_types:
            return results
        else:
            raise Exception(f'{metric_name} is not supported !')
    
    def transform_list_list_types(self, results):
        return [item for sublist in results for item in sublist]
    
    def transform_list_dict_types(self, results):
        return [item for subdict in results for item in subdict.values()]


class ResultsLoader:
    def __init__(self, csv_dir_path):
        self.csv_dir_path = csv_dir_path
        self.method_names = set()
        self.model_names = set()
        self.dataset_names = set()
        self.metric_names = set()
        self.filenames = set()
        for filename in filter(lambda name: '.csv' in name, os.listdir(csv_dir_path)):
            self.filenames.add(filename)
            method_name, model_name, dataset_name, metric_name = filename[:-4].split('_')
            self.method_names.add(method_name)
            self.model_names.add(model_name)
            self.dataset_names.add(dataset_name)
            self.metric_names.add(metric_name)
            self.filenames.add(filename)
        self.results_reader = ResultsReader()
        self.results_transformer = ResutlsTransformer()
            
    def load_file(self, filename, transform = True):
        metric_name = filename[:-4].split('_')[-1]
        with open(self.csv_dir_path + filename) as file:
            results = self.results_reader.read(file, metric_name)
        if transform:
            return self.results_transformer.transform(results, metric_name)
        else:
            return results
    
    def get_results_structure(self, results):
        buffer = ''
        current_type = type(results)
        current_results = results
        while (current_type == list or current_type == dict) and (len(current_results) > 0):
            buffer += f'{current_type}, {len(current_results)} * '
            if current_type == dict:
                current_key, current_results = next(iter(current_results.items()))
                buffer += f'{type(current_key)}:'
            elif current_type == list:
                current_results = current_results[0]
            current_type = type(current_results)

        if current_type != list and current_type != dict:
            buffer += str(current_type)
        else: # empty list of dict case
            buffer += f'{current_type}, empty'

        return buffer
    
    def get_filenames(self, *args):
        if args:
            names = set(args)
            return {filename for filename in self.filenames if names.issubset(filename[:-4].split('_'))}
        else:
            return list(self.filenames)