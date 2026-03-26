"""Steering type registry for convenience defaults.

Maps steering type names to their default dataset paths and alpha ranges.
Individual parameters passed on the command line always override these defaults.
"""

STEERING_CONFIGS = {
    "animal-welfare": {
        "dataset_path": "steering_datasets/animal_welfare_ab.json",
        "eval_hf_dataset": "nmitrani/animal-welfare-prompts",
        "default_alphas": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0",
    },
    "anti-sycophancy": {
        "dataset_path": "steering_datasets/sycophancy_ab.json",
        "eval_hf_dataset": "nmitrani/animal-welfare-prompts",
        "default_alphas": "-1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0",
    },
    "random-direction": {
        "dataset_path": None,  # vectors are pre-generated, no contrastive pairs
        "eval_hf_dataset": "nmitrani/animal-welfare-prompts",
        "default_alphas": "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0",
    },
}
