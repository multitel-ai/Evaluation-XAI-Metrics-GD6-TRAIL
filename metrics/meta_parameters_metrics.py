
import quantus

def meta_param(name, device):

  meta_param_eval = {
      #Faithfullness
          "Faithfulness Correlation": {
              "device": device
          },
          'Faithfulness Estimate': {
             "device": device
          },
          'Pixel-Flipping': {
             "device": device
          },
          'Region Perturbation': {
              "explain_func": quantus.explain, "device": device
          },

        'Monotonicity Arya':  {
           "device": device
          },
        'Monotonicity Nguyen': {
              "device": device
          },
        'Selectivity':{
             "explain_func": quantus.explain,"device": device,
          "segmentation_method": "slic"
          },
        'SensitivityN': {
            "explain_func": quantus.explain, "device": device
        },
        'IROF': {
              "explain_func": quantus.explain, "segmentation_method": "slic", "device": device
        },
  #Localisation
      'Pointing Game':{
		"explain_func": quantus.explain, "device": device
      },
      'Top-K Intersection': {
      "explain_func": quantus.explain, "device": device
      },
      'Relevance Mass Accuracy': {
      "explain_func": quantus.explain, "device": device
      },
       'Relevance Mass Ranking': {
      "explain_func": quantus.explain, "device": device
      },
       'Attribution Localisation':{
         "explain_func": quantus.explain,  "device": device
       },
      'AUC':{
      "explain_func": quantus.explain, "method": "IntegratedGradients", "device": device

      },

  #Randomisation
      'Model Parameter Randomisation':{
       "explain_func": quantus.explain, "method": "Saliency", "device": device
      },

      'Random Logit':{
       "explain_func": quantus.explain, "method": "Saliency", "device": device
      },

  #Robustness
    'Continuity Test':{
       "explain_func": quantus.explain, "device": device
    },
    'Local Lipschitz Estimate': {
        "explain_func": quantus.explain,  "device": device
    },
    'Max-Sensitivity':{
          "explain_func": quantus.explain, "device": device
    },
    'Avg-Sensitivity':{
        "explain_func": quantus.explain,  "device": device
    },

  #Complexity

      "Sparseness": {
    "explain_func": quantus.explain,"device": device
      },
      "Complexity":{
      "explain_func": quantus.explain, "device": device
      },
      "EffectiveComplexity":{
       "explain_func": quantus.explain, "device": device
      },

  #Axiomatic
      "Completeness":{
      "explain_func": quantus.explain,  "device": device
      },
      "Nonsensitivity":{
       "explain_func": quantus.explain,  "device": device
      }

  }
  
  return meta_param_eval[name]


