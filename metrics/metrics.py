import sys
import torch

sys.path.append("Quantus")

import quantus


from .hyper_parameters_metrics import hyper_param_eval
from .meta_parameters_metrics import meta_param

def get_results(model,
                name =  "Faithfulness Correlation",
                x_batch = None,
                y_batch = None,
                a_batch = None,
                perturb_baseline = None,
                xai_method = None,
                device = "cuda"):
    """
    compute a metric for a batch and get results
    :param model: model explained
    :param name: name of the metric to use
    :param x_batch: images batch
    :param y_batch: labels
    :param a_batch: saliency maps
    :param perturb_baseline: perturbation baseline (used by some metrics)
    :param device: device to use for computation
    :return: results of the metric
    """

    metric = {
    #Faithfullness
    "Faithfulness Correlation": quantus.FaithfulnessCorrelation,
    'Faithfulness Estimate': quantus.FaithfulnessEstimate,
    'Pixel-Flipping': quantus.PixelFlipping,
    'Region Perturbation': quantus.RegionPerturbation,
    'Monotonicity Arya': quantus.MonotonicityArya,
    'Monotonicity Nguyen': quantus.MonotonicityNguyen,
    'Selectivity': quantus.Selectivity,
    'SensitivityN': quantus.SensitivityN,
    'IROF': quantus.IterativeRemovalOfFeatures,

    #Localisation
    'Pointing Game': quantus.PointingGame,
    'Top-K Intersection': quantus.TopKIntersection,
    'Relevance Mass Accuracy': quantus.RelevanceMassAccuracy,
    'Relevance Mass Ranking': quantus.RelevanceRankAccuracy,
    'Attribution Localisation': quantus.AttributionLocalisation,
    'AUC': quantus.AUC,

    #Randomisation
    'Model Parameter Randomisation': quantus.ModelParameterRandomisation,
    'Random Logit': quantus.RandomLogit,

    #Robustness
    'Continuity Test': quantus.Continuity,
    'Local Lipschitz Estimate': quantus.LocalLipschitzEstimate,
    'Max-Sensitivity': quantus.MaxSensitivity,
    'Avg-Sensitivity': quantus.AvgSensitivity,

    #Complexity
    "Sparseness": quantus.Sparseness,
    "Complexity": quantus.Complexity,
    "EffectiveComplexity":quantus.EffectiveComplexity,

    #Axiomatic
    "Completeness": quantus.Completeness,
    "Nonsensitivity": quantus.NonSensitivity,
    }

    assert name in metric.keys(), f"Only metrics in {metric.keys()} are allowed!!!"
    #Upsample images if saliency's shape != image's shape
    if a_batch.shape[-2:] != x_batch.shape[-2:]:
        a_batch = torch.nn.functional.interpolate(a_batch, x_batch.shape[-2:], mode='bilinear')

    # get hyperparameters for the metric
    hyper_params = hyper_param_eval[name]

    # Replace the perturabation baseline if specified
    if perturb_baseline is not None:
        hyper_params['perturb_baseline'] = perturb_baseline

    meta_params = meta_param(name, device)

    #Adding the XAI method
    if xai_method is not None:
        meta_params["explain_func"] = xai_method
        
    # Compute and return the metric
    return metric[name](**hyper_params)(model=model,
                                                  x_batch=x_batch.cpu().numpy(),
                                                  y_batch=y_batch.cpu().numpy(),
                                                  a_batch=a_batch.cpu().numpy(),
                                                  **meta_params)