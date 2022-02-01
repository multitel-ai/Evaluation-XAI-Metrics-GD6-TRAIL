import sys
import torch

sys.path.append("Quantus")

import quantus


from .hyper_parameters_metrics import hyper_param_eval
from .meta_parameters_metrics import meta_param

def get_results(model, name =  "Faithfulness Correlation", x_batch = None, y_batch = None, a_batch = None, s_batch = None, device = "cuda"):

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
    "Nonsensitivity": quantus.NonSensitivity

  }

  assert name in metric.keys(), f"Only metrics in {metric.keys()} are allowed!!!"
  
  #Upsample images if saliency's shape != image's shape
  if x_batch.shape[2] != a_batch.shape[2]:
    if torch.cuda.is_available():
      with torch.cuda.device(device if device =="cuda" else None):
        a_batch = torch.nn.Upsample(scale_factor=x_batch.shape[2]/a_batch.shape[2], mode='nearest')(torch.from_numpy(a_batch)).cpu().detach().numpy()
    else:
      a_batch = torch.nn.Upsample(scale_factor=x_batch.shape[2]/a_batch.shape[2], mode='bilinear')(torch.from_numpy(a_batch)).cpu().detach().numpy()
      

  return metric[name](**hyper_param_eval[name])(model = model, x_batch = x_batch, y_batch = y_batch,
    a_batch = a_batch, **meta_param(name, device))