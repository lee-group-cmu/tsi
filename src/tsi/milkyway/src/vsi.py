import os
import dill
import pandas as pd
import numpy as np
import torch
from typing import Optional, Union, Dict, List, Tuple, Any
import random
from catboost import CatBoostClassifier

from utils import setup_paths, get_eval_grid, get_prior, get_estimator, create_test_statistic
from lf2i.inference import LF2I
from lf2i.utils.other_methods import hpd_region
from lf2i.diagnostics.coverage_probability import compute_indicators_posterior

class VSI:
    """
    Valid Scientific Inference (VSI) wrapper for LF2I workflows.

    Provides a structured interface for setting up simulation-based inference experiments
    using LF2I test statistics, confidence set inference, diagnostics, and plotting data preparation.
    """
    def __init__(self,
                 confidence_level: float = 0.90,
                 evaluation_grid_size: int = 1_000_000,
                 prior_method: str = "kde",
                 estimator_method: str = "fmpe",  # for posterior
                 test_statistic_method: str = "posterior",  # or waldo
                 calibration_method: str = "critical-values",
                 seed: int = 42,
                 notebook: bool = False
                 ) -> None:
        
        self.confidence_level = confidence_level
        self.evaluation_grid_size = evaluation_grid_size
        self.prior_method = prior_method
        self.estimator_method = estimator_method
        self.test_statistic_method = test_statistic_method
        self.calibration_method = calibration_method
        self.seed = seed

        self.filepaths = setup_paths(notebook=notebook)

        self.data = {}
        self.parameter_names = None
        self.sample_dim = None
        self.evaluation_grid = None
        self.prior = None
        self.estimator = None
        self.test_statistic = None
        self.lf2i = None

        self.inference_lf2i_info = None
        self.inference_estimator_info = None

        self.diagnostics_lf2i_info = None
        self.diagnostics_estimator_info = None
        self.mean_proba_posterior_classifier = None

    def load_data(self,
              X_train: torch.Tensor, y_train: torch.Tensor,
              X_calibration: torch.Tensor, y_calibration: torch.Tensor,
              X_test: Optional[torch.Tensor] = None, y_test: Optional[torch.Tensor] = None,
              parameter_names: Optional[List[str]] = None
             ) -> None:
        """
        Load data into VSI and update sample dimension and parameter names.

        Parameters
        ----------
        X_train, y_train : torch.Tensor
            Training inputs and outputs.
        X_calibration, y_calibration : torch.Tensor
            Calibration inputs and outputs.
        X_test, y_test : torch.Tensor, optional
            Test inputs and outputs.
        parameter_names : list of str, optional
            Names for the output parameters. If not provided, default names will be used.
        """
        self.data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_calibration": X_calibration,
            "y_calibration": y_calibration,
            "X_test": X_test,
            "y_test": y_test
        }

        # Set sample_dim from input features
        if X_train.ndim != 2:
            raise ValueError(f"Expected X_train to be a 2D tensor, got shape {X_train.shape}")
        self.sample_dim = X_train.shape[1]

        # Set parameter names from output dimension
        if y_train.ndim != 2:
            raise ValueError(f"Expected y_train to be a 2D tensor, got shape {y_train.shape}")

        num_outputs = y_train.shape[1]
        if parameter_names is not None:
            if len(parameter_names) != num_outputs:
                raise ValueError(f"Length of parameter_names ({len(parameter_names)}) must match y_train.shape[1] ({num_outputs})")
            self.parameter_names = parameter_names
        else:
            self.parameter_names = [f"param_{i}" for i in range(num_outputs)]

    def generate_evaluation_grid(self, data: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if data is None:
            if "y_calibration" in self.data and self.data["y_calibration"] is not None:
                data = self.data["y_calibration"]
            else:
                raise ValueError("No evaluation data provided and no default y_calibration available.")
        grid_size = kwargs.pop("evaluation_grid_size", self.evaluation_grid_size)
        grid = get_eval_grid(data, evaluation_grid_size=grid_size, **kwargs)
        self.evaluation_grid = torch.tensor(grid, dtype=torch.float32)
        return self.evaluation_grid

    def set_prior(self, 
                 prior_samples: Optional[Union[np.ndarray, torch.Tensor]] = None, 
                 prior_method: Optional[Union[str, callable]] = None,
                 prior_args: Optional[dict] = None,
                 **kwargs) -> torch.distributions.Distribution:
        if prior_samples is None:
            if "y_train" in self.data and self.data["y_train"] is not None:
                prior_samples = self.data["y_train"]
            else:
                raise ValueError("No prior_samples provided and no default y_train available in data.")
        if prior_method is None:
            prior_method = self.prior_method
        if prior_args is None:
            prior_args = {}
        self.prior = get_prior(prior_samples=prior_samples, 
                               prior_method=prior_method, 
                               prior_args=prior_args, 
                               **kwargs)
        return self.prior

    def train_estimator(self, **kwargs) -> torch.distributions.Distribution:
        if "y_train" not in self.data or self.data["y_train"] is None:
            raise ValueError("Training labels (y_train) are not loaded.")
        if "X_train" not in self.data or self.data["X_train"] is None:
            raise ValueError("Training inputs (X_train) are not loaded.")
        if self.prior is None:
            raise ValueError("Prior is not set. Please run set_prior() before training the estimator.")
        self.estimator = get_estimator(
            parameters=self.data["y_train"],
            samples=self.data["X_train"],
            prior=self.prior,
            estimator_type=self.estimator_method,
            **kwargs
        )
        return self.estimator

    def build_test_statistic(self, test_statistic: str = None, **kwargs):
        if test_statistic is None:
            if self.test_statistic_method is None:
                raise ValueError("No test statistic type provided or set in self.test_statistic_method.")
            test_statistic = self.test_statistic_method
        poi_dim = kwargs.pop("poi_dim", None)
        if poi_dim is None:
            if "y_train" in self.data and self.data["y_train"] is not None:
                poi_dim = self.data["y_train"].shape[1]
            else:
                raise ValueError("Cannot infer poi_dim. Provide 'poi_dim' in kwargs or load y_train data.")
        estimator_arg = self.estimator if self.estimator is not None else self.estimator_method
        ts = create_test_statistic(test_statistic, self.prior, estimator_arg, poi_dim,
                                   evaluation_grid=self.evaluation_grid, **kwargs)
        self.test_statistic = ts
        return ts

    def fit_lf2i(self, **test_statistic_kwargs):
        if self.test_statistic is None:
            raise ValueError("Test statistic not set. Please build the test statistic first using build_test_statistic().")
        self.lf2i = LF2I(test_statistic=self.test_statistic, **test_statistic_kwargs)
        return self.lf2i

    def inference_lf2i(self, **inference_kwargs):
        if self.lf2i is None:
            raise ValueError("LF2I model is not instantiated. Please run fit_lf2i() first.")
        if "x" not in inference_kwargs:
            if "X_test" not in self.data or self.data["X_test"] is None:
                raise ValueError("Test inputs (X_test) are not loaded.")
            inference_kwargs["x"] = self.data["X_test"]
        if "evaluation_grid" not in inference_kwargs:
            if self.evaluation_grid is None:
                raise ValueError("Evaluation grid is not set. Please generate an evaluation grid first.")
            inference_kwargs["evaluation_grid"] = self.evaluation_grid
        if "confidence_level" not in inference_kwargs:
            inference_kwargs["confidence_level"] = self.confidence_level
        if "calibration_method" not in inference_kwargs:
            inference_kwargs["calibration_method"] = self.calibration_method
        if "calibration_model" not in inference_kwargs:
            inference_kwargs["calibration_model"] = 'cat-gb'
        if "calibration_model_kwargs" not in inference_kwargs:
            inference_kwargs["calibration_model_kwargs"] = {
                'cv': {'iterations': [10, 25, 50, 100, 250, 300, 500, 1000, 1250, 1500], 
                       'depth': [1, 2, 3, 4, 5, 10],
                       'l2_leaf_reg': [3, 5, 10, 30],
                       'learning_rate': [0.01, 0.05, 0.1, 0.005],
                       'bagging_temperature': [0, 1, 3, 5]},
                'n_iter': 500
            }
        if "T" not in inference_kwargs:
            inference_kwargs["T"] = (self.data["y_train"], self.data["X_train"])
        if "T_prime" not in inference_kwargs:
            inference_kwargs["T_prime"] = (self.data["y_calibration"], self.data["X_calibration"])
        confidence_sets = self.lf2i.inference(**inference_kwargs)
        self.inference_lf2i_info = confidence_sets
        return confidence_sets

    def inference_estimator(self, **inference_kwargs):
        if self.lf2i is None:
            raise ValueError("LF2I model is not instantiated. Please run fit_lf2i() first.")
        if "parameters" not in inference_kwargs:
            if "y_test" not in self.data or self.data["y_test"] is None:
                raise ValueError("Test inputs (y_test) are not loaded.")
            inference_kwargs["parameters"] = self.data["y_test"]
        if "samples" not in inference_kwargs:
            if "X_test" not in self.data or self.data["X_test"] is None:
                raise ValueError("Test inputs (X_test) are not loaded.")
            inference_kwargs["samples"] = self.data["X_test"]
        if "parameter_grid" not in inference_kwargs:
            if self.evaluation_grid is None:
                raise ValueError("Evaluation grid is not set. Please generate an evaluation grid first.")
            inference_kwargs["parameter_grid"] = self.evaluation_grid
        if "credible_level" not in inference_kwargs:
            inference_kwargs["credible_level"] = self.confidence_level
        if "param_dim" not in inference_kwargs:
            inference_kwargs["param_dim"] = len(self.parameter_names)
        if "return_credible_regions" not in inference_kwargs:
            inference_kwargs["return_credible_regions"] = True
        if "num_level_sets" not in inference_kwargs:
            inference_kwargs["num_level_sets"] = 10_000
        if "batch_size" not in inference_kwargs:
            inference_kwargs["batch_size"] = 1
        
        # default will return credible regions and indicators (for diagnostics)
        inference_estimator = compute_indicators_posterior(posterior=self.lf2i.test_statistic.estimator,
                                                           **inference_kwargs)
        self.inference_estimator_info = inference_estimator
        return inference_estimator

    def diagnostics_lf2i(self, **kwargs):
        if self.lf2i is None:
            raise ValueError("LF2I model is not instantiated. Please run fit_lf2i() first.")
        if "T_double_prime" not in kwargs:
            kwargs["T_double_prime"] = (self.data["y_test"], self.data["X_test"])
        if "coverage_estimator" not in kwargs:
            kwargs["coverage_estimator"] = "cat-gb"
        kwargs.setdefault("calibration_method", self.calibration_method)
        kwargs.setdefault("confidence_level", self.confidence_level)
        diag = self.lf2i.diagnostics(region_type="lf2i", **kwargs)
        self.diagnostics_lf2i_info = diag
        return diag

    def diagnostics_estimator(self, **kwargs):
        if self.lf2i is None:
            raise ValueError("LF2I model is not instantiated. Please run fit_lf2i() first.")
        if "indicators" not in kwargs:
            raise ValueError("For posterior diagnostics, please supply 'indicators' in kwargs.")
        if "parameters" not in kwargs:
            kwargs["parameters"] = self.data["y_test"]
        if "coverage_estimator" not in kwargs:
            kwargs["coverage_estimator"] = "cat-gb"
        kwargs.setdefault("calibration_method", self.calibration_method)
        kwargs.setdefault("confidence_level", self.confidence_level)
        diag = self.lf2i.diagnostics(region_type="posterior", **kwargs)
        self.diagnostics_estimator_info = diag
        return diag
    
    def classifier_mean_proba_estimator(self, diagnostics_info=None, random_seed=42):
        """
        Train a CatBoost classifier on diagnostic indicators and compute mean posterior probabilities.
        """
        if self.diagnostics_estimator_info is None and diagnostics_info is None:
            raise ValueError("Please provide or run estimator diagnostics first.")
        elif diagnostics_info is None:
            diagnostics_info = self.diagnostics_estimator_info

        # Data setup
        X_classifier = np.array(self.data["y_test"])
        y_classifier, _ = self.inference_estimator_info

        # Train classifier
        classifier = CatBoostClassifier(random_seed=random_seed, verbose=0)
        classifier.fit(X_classifier, y_classifier)

        mean_proba_posterior = classifier.predict_proba(X_classifier)[:, 1]

        self.mean_proba_posterior_classifier = mean_proba_posterior

        return mean_proba_posterior


    def prepare_plot_data(self) -> Dict[str, pd.DataFrame]:
        output = {}
        if self.inference_lf2i_info is not None:
            cs = self.inference_lf2i_info
            if isinstance(cs, torch.Tensor):
                cs = cs.numpy()
            if isinstance(cs, list):
                cs = np.array(cs)
            num_params = cs.shape[1]
            columns = [f"param_{i}" for i in range(num_params)]
            output["confidence_sets"] = pd.DataFrame(cs, columns=columns)
        if self.inference_estimator_info is not None:
            hpd = self.inference_estimator_info.get("hpd_params")
            if hpd is not None:
                if isinstance(hpd, torch.Tensor):
                    hpd = hpd.numpy()
                num_params = hpd.shape[1]
                columns = [f"param_{i}" for i in range(num_params)]
                output["hpd"] = pd.DataFrame(hpd, columns=columns)
            achieved = self.inference_estimator_info.get("achieved_level")
            output["hpd_achieved_level"] = achieved
        output["diagnostics"] = pd.DataFrame()
        return output

    def find_nearest_input(self,
                        target_params: Optional[List[float]] = None,
                        parameter_names: Optional[List[str]] = None
                        ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Find the test input (X) whose corresponding label (y) is closest to a given set of target values.
        If no target values are given, randomly select a test point.

        Parameters
        ----------
        target_params : list of float, optional
            Target parameter values to search for (e.g., [logg, teff, fe_h]).
        parameter_names : list of str, optional
            Names of the label dimensions to match on (must be a subset of self.label_names).
            If not provided, all label dimensions are used.

        Returns
        -------
        x_selected : torch.Tensor
            Input vector with shape (1, n_features).
        index : int
            Index in the test set.
        y_selected : torch.Tensor
            Corresponding label vector with shape (1, n_labels).
        """
        if "X_test" not in self.data or self.data["X_test"] is None:
            raise ValueError("Test inputs (X_test) are not loaded.")
        if "y_test" not in self.data or self.data["y_test"] is None:
            raise ValueError("Test labels (y_test) are not loaded.")
        if not hasattr(self, "parameter_names") or self.parameter_names is None:
            raise ValueError("VSI object must define self.label_names corresponding to y_test columns.")

        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        if target_params is None:
            index = random.randint(0, len(X_test) - 1)
            x_selected = X_test[index].unsqueeze(0)
            y_selected = y_test[index].unsqueeze(0)
            return x_selected, index, y_selected

        if parameter_names is None:
            # Use all parameters
            y_subset = y_test
            target = torch.tensor(target_params, dtype=torch.float32).view(1, -1)
        else:
            # Check and find indices for selected parameters
            param_indices = []
            for name in parameter_names:
                name_upper = name.upper()
                if name_upper not in [ln.upper() for ln in self.parameter_names]:
                    raise ValueError(f"Parameter '{name}' not found in label_names: {self.parameter_names}")
                param_indices.append([ln.upper() for ln in self.parameter_names].index(name_upper))

            y_subset = y_test[:, param_indices]
            target = torch.tensor(target_params, dtype=torch.float32).view(1, -1)

        distances = torch.norm(y_subset - target, dim=1)
        index = torch.argmin(distances).item()

        x_selected = X_test[index].unsqueeze(0)
        y_selected = y_test[index].unsqueeze(0)
        return x_selected, index, y_selected

    def save(self, save_inference = False, extra_objects: dict = None):
        assets_dir = self.filepaths.get("assets", ".")
        if self.test_statistic_method is not None:
            test_stat = self.test_statistic_method
        elif self.test_statistic is not None:
            test_stat = type(self.test_statistic).__name__
        else:
            test_stat = "no_test_stat"
        folder_name = f"VSI_conf{self.confidence_level}_grid{self.evaluation_grid_size}_est{self.estimator_method}_test{test_stat}"
        folder_path = os.path.join(assets_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        def get_unique_filename(base_path, filename):
            file_path = os.path.join(base_path, filename)
            if not os.path.exists(file_path):
                return file_path
            version = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(os.path.join(base_path, f"{base}_V{version}{ext}")):
                version += 1
            return os.path.join(base_path, f"{base}_V{version}{ext}")
        main_filename = "VSI_instance.pkl"
        main_file = get_unique_filename(folder_path, main_filename)
        with open(main_file, "wb") as f:
            dill.dump(self, f)
        print(f"Saved VSI instance to {main_file}")
        if extra_objects is not None:
            for key, obj in extra_objects.items():
                extra_filename = f"VSI_{key}.pkl"
                extra_file = get_unique_filename(folder_path, extra_filename)
                with open(extra_file, "wb") as f:
                    dill.dump(obj, f)
                print(f"Saved extra object '{key}' to {extra_file}")
        if save_inference: # need to add controls for when inference attr empty
            info_filename = "VSI_inference_lf2i_info.pkl"
            info_file = get_unique_filename(folder_path, info_filename)
            with open(info_file, "wb") as f:
                dill.dump(self.inference_lf2i_info, f)
            print(f"Saved inference_lf2i_info to {info_file}")
        if save_inference: # need to add controls for when inference attr empty
            info_filename = "VSI_inference_estimator_info.pkl"
            info_file = get_unique_filename(folder_path, info_filename)
            with open(info_file, "wb") as f:
                dill.dump(self.inference_estimator_info, f)
            print(f"Saved inference_estimator_info to {info_file}")