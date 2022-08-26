
import quantus


def get_hyper_param_eval(img_size=224, num_classes=1000):
    if img_size == 32:
        small_patch_size = 2
        big_patch_size = 12
        num_classes = 10
    else:
        small_patch_size = 14
        big_patch_size = 28
        num_classes = 1000


    hyper_param_eval = {
        #Faithfullness
            "Faithfulness Correlation": {
                "nr_runs": 100,
                "subset_size": img_size,
                "perturb_baseline": "black",
                "perturb_func": quantus.baseline_replacement_by_indices,
                "similarity_func": quantus.correlation_pearson,
                "abs": False,
                "return_aggregate": False,
            },
            'Faithfulness Estimate': {
                "similarity_func": quantus.correlation_pearson,
                 "perturb_func": quantus.baseline_replacement_by_indices,
                "img_size": img_size,
                "features_in_step": img_size,
                "perturb_baseline": "black",
                "pixels_in_step": big_patch_size,
            },
            'Pixel-Flipping': {
                "features_in_step": img_size,
                "perturb_baseline": "black",
                "perturb_func": quantus.baseline_replacement_by_indices,
            },
            'Region Perturbation': {
                "patch_size": big_patch_size,
                "regions_evaluation": 100,
                "img_size": img_size,
                "random_order": False,
                "perturb_func": quantus.baseline_replacement_by_patch,
                "perturb_baseline": "uniform",
            },

          'Monotonicity Arya':  {
                "features_in_step": img_size,
                "perturb_baseline": "black",
                "perturb_func": quantus.baseline_replacement_by_indices,
                "similarity_func": quantus.correlation_spearman
            },
          'Monotonicity Nguyen': {
                "nr_samples": 10,
                "features_in_step": img_size,
                "perturb_baseline": "uniform",
                "perturb_func": quantus.baseline_replacement_by_indices,
            "similarity_func": quantus.correlation_spearman,
            },
          'Selectivity':{
                "patch_size": small_patch_size,
                "perturb_func": quantus.baseline_replacement_by_patch,
                "perturb_baseline": "black",
            },
          'SensitivityN': {
                "features_in_step": big_patch_size,
                "n_max_percentage": 0.8,
                "img_size": img_size,
                "similarity_func": quantus.correlation_pearson,
                "perturb_func": quantus.baseline_replacement_by_indices,
                "perturb_baseline": "uniform",
          },
          'IROF': {
                "segmentation_method": "slic",
                "perturb_baseline": "mean",
                "perturb_func": quantus.baseline_replacement_by_indices,
          },
        #Localisation
            'Pointing Game':{

            },
            'Top-K Intersection': {

            },
            'Relevance Mass Accuracy': {

            },
             'Relevance Mass Ranking': {
            },
             'Attribution Localisation':{
             },
            'AUC':{

            },

        #Randomisation
            'Model Parameter Randomisation':{
                 "layer_order": "top_down",
                "similarity_func": quantus.correlation_spearman,
                "normalize": True,
            },

            'Random Logit':{
                "num_classes": num_classes,
                "similarity_func": quantus.ssim,
            },

        #Robustness
          'Continuity Test':{
             "nr_patches": 4,
            "nr_steps": 10,
            "img_size": img_size,
            "perturb_baseline": "black",
            "similarity_func": quantus.correlation_spearman,
            "perturb_func": quantus.translation_x_direction,
          },
          'Local Lipschitz Estimate': {
              "nr_samples": 10,
                "perturb_std": 0.1,
                "perturb_mean": 0.1,
                "norm_numerator": quantus.distance_euclidean,
                "norm_denominator": quantus.distance_euclidean,
                "perturb_func": quantus.gaussian_noise,
                "similarity_func": quantus.lipschitz_constant,
          },
          'Max-Sensitivity':{
                "nr_samples": 10,
                "perturb_radius": 0.2,
                "norm_numerator": quantus.fro_norm,
                "norm_denominator": quantus.fro_norm,
                "perturb_func": quantus.uniform_sampling,
                "similarity_func": quantus.difference,
          },
          'Avg-Sensitivity':{
                "nr_samples": 10,
                "perturb_radius": 0.2,
                "norm_numerator": quantus.fro_norm,
                "norm_denominator": quantus.fro_norm,
                "perturb_func": quantus.uniform_sampling,
                "similarity_func": quantus.difference,
          },

        #Complexity

            "Sparseness": {

            },
            "Complexity":{

            },
            "EffectiveComplexity":{
                "eps": 1e-5,
            },

        #Axiomatic
            "Completeness":{
                "abs": False,
                "disable_warings": True,
            },
            "Nonsensitivity":{
                "abs": True,
                "eps": 1e-5,
                "n_samples": 10,
                "perturb_baseline": "black",
                "perturb_func": quantus.baseline_replacement_by_indices,
            }

        }
    return hyper_param_eval


metric_types = {
    
    "robustness": ['Local Lipschitz Estimate', 'Avg-Sensitivity', "Max-Sensitivity", 'Continuity Test'],
    "randomisation": ['Model Parameter Randomisation',  'Random Logit']
}

