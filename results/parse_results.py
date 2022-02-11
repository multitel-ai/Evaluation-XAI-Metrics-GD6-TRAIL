import pandas as pd
import numpy as np
import json
import yaml
import scikit_posthocs as sp
from scipy import stats
from matplotlib import pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks

def compute_plot_Nemenyi(data, filename, model ="resnet50", dataset = "imagenet"):
    ranks = np.array([rankdata(-p) for p in data.values])
    average_ranks = np.mean(ranks, axis=0)
    cd = compute_CD(average_ranks,
                    n=data.shape[0],
                    alpha='0.05',
                    test='nemenyi')
    # This method generates the plot.
    print("Go and compute!!!")
    graph_ranks(average_ranks,
                names=list(data),
                cd=cd,
                width=10,
                textspace=2,
                reverse=True)

    print("Savging ", filename)

    plt.savefig(f'figures/rank_per_metric/{filename}_{model}_{dataset}.png')
    print('Saved!!!')
def parser_method_dict(df, batch = 16):
    ''''This function parse the dict and then compute the AUC per image
        It returns a dataframe of the shape (Batch, Nb_images_per_bach)
        np.trapz integrates the function under the curve
    '''
    dataf = pd.DataFrame(columns = [f"{i}" for i in range(batch)])
    for i in range(df.shape[0]):
        row = yaml.safe_load(df.iloc[i].iloc[0])
        dataf.loc[i] = [np.trapz(row[j]) for j in row]
    #print(dataf)
    return dataf

def parser_method_dict_with_layers(df, batch = 16):
    ''''This function parse the dict and then compute the AUC per image
        It returns a dataframe of the shape (Nb_Batch, Nb_images_per_bach)
        np.trapz integrates the function under the curve
        Note that df.iloc[0] has a dic e.g., {"layer1": [list of corr of bacth], "layer2": [list of corr of batch]}
    '''
    dataf = pd.DataFrame(columns = [f"{i}" for i in range(batch)])
    for i in range(df.shape[0]): # loop over the number of batches
        row = yaml.safe_load(df.iloc[i].iloc[0]) # Get the batch result
        
        # p is the index of image, j is the layer_name, np.trapz computes the AUC
        dataf.loc[i] = [np.trapz([row[j][p] for j in row]) for p in range(batch)]
    return dataf

metrics =  ['Monotonicity Nguyen',  'Local Lipschitz Estimate',
            'Faithfulness Estimate', 'Faithfulness Correlation', 
            'Avg-Sensitivity', 'Random Logit',
            'Max-Sensitivity', 'Sparseness', 
            'EffectiveComplexity',  'Monotonicity Arya',
             'Complexity', 'Pixel-Flipping',
            "Selectivity", 'Model Parameter Randomisation'] 
                # ['SensitivityN': problem with implementation,
                #'Region Perturbation' seems to be same as region perturbation, 
                #'Continuity Test': Difficult to aggregate result, the paper just plot it
                #'Completeness' always returns False]
                # Nonsentitivity is removed
    
transform = {'Monotonicity Nguyen': lambda x: x, 'Local Lipschitz Estimate': lambda x: -x, 
            'Faithfulness Estimate': abs, 'Faithfulness Correlation': abs, 
            'Avg-Sensitivity': lambda x: -x, 'Random Logit': lambda x: x,
             'Sparseness': lambda x: x, 'EffectiveComplexity': lambda x: -x,
             'Nonsensitivity': lambda x: -x, 'Pixel-Flipping': lambda x: x.apply(lambda row: - np.trapz(row), axis=1),
             'Max-Sensitivity': lambda x: -x, 'Complexity': lambda x: -x, 
             "Selectivity": lambda x: -parser_method_dict(x), 'Model Parameter Randomisation': lambda x: parser_method_dict_with_layers(x),
             'Monotonicity Arya': lambda x: x,
            }
    
methods = ['integratedgrad', 'smoothgrad', 'guidedbackprop', 'rise', 'gradcam', 'scorecam', 'layercam', 'random', 'sobel', "gaussian", "polycam"]

if __name__ == '__main__':

    model = "resnet50"

    dico_ranks = {}
    alpha = 0.05 #Seuil de significativité

    group_nemenyi = {}

    dataset = "imagenet"
    for metr in metrics:
        data_nemenyi = pd.DataFrame()
        print("-- Metric: ", metr)
        for meth in methods:
            csv_name = f"../csv/{meth}_{model}_{dataset}_{metr}.csv"
            df = pd.read_csv(csv_name, header = None)
            arr_values = transform[metr](df).values.flatten()
            pd.DataFrame(arr_values, columns =["value_per_image"]).to_csv(f"parsed_csv/processed_{meth}_{model}_{dataset}_{metr}")
            data_nemenyi[meth] = arr_values

        #Draw Nemenyi ranks
        #compute_plot_Nemenyi(data= data_nemenyi, filename= metr)
        # ranks = np.array([rankdata(-p) for p in data_nemenyi.values])
        # average_ranks = np.mean(ranks, axis=0)
        
        # dico_ranks[metr] = average_ranks/len(metrics)
        
        # result = sp.posthoc_nemenyi_friedman(data_nemenyi)