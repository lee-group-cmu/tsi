import torch 
import numpy as np
import raw_events as rev
from typing import List, Callable, Union
from dataclasses import dataclass
import nde_models
import copy
import cr_sources as crs
from sbi.neural_nets.estimators.score_estimator import VPScoreEstimator

root_dir = "/home/export/ajshen/ada-sbi-cosmic-rays/"

@dataclass
class DataConfig:
    manifest_filename: str
    filtered_manifest_filename: str
    data_dir: str 
   
    manifest_generator: Callable[[str, str], None]
    
    train_sample_ratio: float 
    holdout_of_train_ratio: float
    val_of_train_ratio: float 
    cal_sample_ratio: float 
    test_sample_ratio: float
    
    splits_dir: str

DC_TRAIN_IS_TEST = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_filtered_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        reference_ratio=0.7,
        train_carveout_ratio=0.6,
        train_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        test_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        astropy_names=["crab", "crab"],
        observer_latitude=-20,
        trajectory_azimuth_epsilon_deg_per_source=[5, 5],
        energy_bins=100,
        user_min_log10_energy_gev=1.5,
        user_max_log10_energy_gev=5,
    ),
    
    train_sample_ratio=0.5,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.2,
    cal_sample_ratio=0.1,
    test_sample_ratio=0.3,
    splits_dir="vsi_splits/"
)

DC_TRAIN_IS_TEST_ENERGY_ONLY = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_energy_only.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_energy_prior_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=1.5,
        max_log10_energy_gev=5,
        train_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        test_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        astropy_names=["crab", "mrk421"],
        train_ratio=0.4,
        test_ratio=0.2,
        energy_bins=100,
        zenith_cutoffs=False
    ),
    
    train_sample_ratio=0.1,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.1,
    cal_sample_ratio=0.01,
    test_sample_ratio=0.01,
    splits_dir="vsi_splits_energy_only/"
)

DC_PRIOR_SHIFT_ENERGY_ZENITH = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_energy_only.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_energy_prior_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=1.5,
        max_log10_energy_gev=6,
        train_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        test_log10flux_of_log10energy_gev=rev.log10_mrk421_flux,
        astropy_names=["crab", "mrk421"],
        train_ratio=0.5,
        test_ratio=0.3,
        energy_bins=100,
        zenith_cutoffs=True,
        observer_latitude=-20,
    ),
    
    train_sample_ratio=0.4,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.1,
    cal_sample_ratio=0.01,
    test_sample_ratio=1.0,
    splits_dir="vsi_splits_energy_zenith/"
)

DC_TRIMMED_PRIORS = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_trimmed_priors.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_all_prior_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=3,
        max_log10_energy_gev=6,
        min_zenith=0.2,
        train_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        test_log10flux_of_log10energy_gev=rev.log10_mrk421_flux,
        astropy_names=["crab", "mrk421"],
        cal_sample_ratio=1.0,
        train_ratio=0.75,
        test_ratio=1.0,
        num_calibration_bins=10,
        num_train_test_bins=200,
        observer_latitude=-15,
        minimum_weight=1e-10,
        minimum_features=10
    ),
    
    train_sample_ratio=1.0,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.1,
    cal_sample_ratio=1.0,
    test_sample_ratio=1.0,
    splits_dir="vsi_splits_trimmed_priors/"
)

DC_FULL_PRIORS = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_full_priors.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_all_prior_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=3,
        max_log10_energy_gev=6,
        min_zenith=0,
        train_log10flux_of_log10energy_gev=rev.log10_crab_flux,
        test_log10flux_of_log10energy_gev=rev.log10_mrk421_flux,
        astropy_names=["crab", "mrk421"],
        cal_sample_ratio=1.0,
        train_ratio=0.7,
        test_ratio=1.0,
        num_calibration_bins=10,
        num_train_test_bins=200,
        observer_latitude=-15,
        minimum_weight=1e-12,
        minimum_features=10,
        target_calibration_size=100_000
    ),
    
    train_sample_ratio=1.0,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.1,
    cal_sample_ratio=1.0,
    test_sample_ratio=1.0,
    splits_dir="vsi_splits_full_priors/"
)

DC_FLEXIBLE_PRIORS = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_flexible_priors.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_flexible_manifest(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=3,
        max_log10_energy_gev=6,
        astropy_names=["crab", "mrk421"],
        cal_sample_ratio=1.0,
        train_ratio=0.7,
        test_ratio=1.0,
        num_calibration_bins=10,
        minimum_features=10,
        target_calibration_size=100_000
    ),
    
    train_sample_ratio=1.0,
    holdout_of_train_ratio=0.1,
    val_of_train_ratio=0.1,
    cal_sample_ratio=1.0,
    test_sample_ratio=1.0,
    splits_dir="vsi_splits_flexible_priors/"
)

DC_FLEXIBLE_PRIORS_UNIFORM_TEST = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_flexible_priors_uniform_test.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_flexible_manifest_uniform_test(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=3,
        max_log10_energy_gev=6,
        test_ratio=0.3,
        num_uniform_bins=10,
        minimum_features=10,
        target_cal_test_size=150_000,
        caltest_high_energy_sample_ratio=0.95
    ),
    
    train_sample_ratio=1.0,
    holdout_of_train_ratio=0,
    val_of_train_ratio=0.1,
    cal_sample_ratio=1.0,
    test_sample_ratio=1.0,
    splits_dir=f"{root_dir}vsi_splits_flexible_priors_uniform_test/"
)

DC_FLEXIBLE_PRIORS_UNIFORM_TEST_4GEV = DataConfig(
    manifest_filename="vsi_all_manifest.pkl",
    filtered_manifest_filename="vsi_all_manifest_filtered_flexible_priors_uniform_test_4gev.pkl",
    data_dir="vsi_all/",
    
    manifest_generator=lambda m, f: rev.generate_flexible_manifest_uniform_test(
        manifest_filename=m,
        filtered_manifest_filename=f,
        min_log10_energy_gev=3,
        max_log10_energy_gev=4,
        test_ratio=0.3,
        num_uniform_bins=10,
        minimum_features=10,
        target_cal_test_size=150_000,
        caltest_high_energy_sample_ratio=0.95
    ),
    
    train_sample_ratio=1.0,
    holdout_of_train_ratio=0,
    val_of_train_ratio=0.1,
    cal_sample_ratio=1.0,
    test_sample_ratio=1.0,
    splits_dir=f"{root_dir}vsi_splits_flexible_priors_uniform_test_4gev/"
)

dc_config_dict = {
    "train_is_test": DC_TRAIN_IS_TEST,
    "energy_only": DC_TRAIN_IS_TEST_ENERGY_ONLY,
    "energy_zenith": DC_PRIOR_SHIFT_ENERGY_ZENITH,
    "trimmed_priors": DC_TRIMMED_PRIORS,
    "full_priors": DC_FULL_PRIORS,
    "flexible_priors": DC_FLEXIBLE_PRIORS,
    "flexible_priors_uniform_test": DC_FLEXIBLE_PRIORS_UNIFORM_TEST,
    "flexible_priors_uniform_test_4gev": DC_FLEXIBLE_PRIORS_UNIFORM_TEST_4GEV
}

@dataclass
class ExperimentConfig:
    seed: int
    gpu_index: int
    source_data_config: DataConfig
    detector_layout: torch.Tensor
    max_shift_radius: int
    downsample_factor: int
    min_features_threshold: int
    time_averages: List[bool]
    
    model_constructor: Callable[[torch.device, int, List[int]], Union[torch.nn.Module, nde_models.WeightedNPSE]]
    model_save_path: str

    train_data_pkl: str
    val_data_pkl: str
    train_param_mins: torch.Tensor 
    train_param_maxes: torch.Tensor
    train_batch_size: int
    learning_rate: float
    val_epoch_cooldown: int
    max_epochs: int
    train_limit_batches_per_epoch: int
    train_error_estimate_batches: int
    
    eval_data_pkl: str
    eval_param_mins: torch.Tensor 
    eval_param_maxes: torch.Tensor
    coverage_calc_grid_num_points: int
    eval_limit_count: int
    confidence_levels: List[float]
    
    calibration_batch_limit_count: int
    calibration_num_posterior_samples: int
    calibration_data_pkl: str
    cal_loader_repeats: int = 1
    cal_limit_zenith_to_train: bool = False
    
    loss2_ratio: float = 1.0
    use_existing_trained_model: bool = True
    additional_calibration_data_pkl: str = None
    train_ds_on_cpu: bool = False
    train_sample_ratio: float = 1.0
    val_sample_ratio: float = 1.0
    cal_sample_ratio: float = 1.0
    restrict_azimuth_to_pm90deg: bool = False
    
    train_differential_flux: Callable[[float], float] = None
    train_astropy_source_name: str = "crab"
    train_observer_latitude: float = -15.0
    train_higher_energy_prior_config: dict = None
    
    no_azimuth: bool = False
    
    use_sbi: bool = False
    use_posterior: bool = False
    no_train_weights: bool = False
    max_sbi_energy: float = 99.0
    
    use_subsampling: bool = False

DEFAULT_CONFIG = ExperimentConfig(
    seed=0,
    gpu_index=0,
    source_data_config=DC_TRAIN_IS_TEST,
    detector_layout=torch.ones(2000, 2000),
    max_shift_radius=800,
    downsample_factor=20,
    min_features_threshold=10,
    time_averages=[True, False, False],
    
    model_constructor= lambda d, num_channels, fgs: nde_models.JointCRNF(
        device=d,
        context_model=nde_models.ContextModel(
            device=d,
            num_channels=num_channels,
            final_grid_shape=fgs,
            context_size=256,
            kernel_size=13
        ),
        num_flows=4
    ),
    model_save_path="default_joint.pt",
    
    train_data_pkl="train.pkl",
    val_data_pkl="val.pkl",
    train_param_mins=torch.tensor([1.5, np.deg2rad(42), np.deg2rad(-60)], dtype=torch.float32),
    train_param_maxes=torch.tensor([5.5, np.deg2rad(65), np.deg2rad(60)], dtype=torch.float32),
    train_batch_size=2048,
    learning_rate=1e-3,
    val_epoch_cooldown=10,
    max_epochs=200,
    train_limit_batches_per_epoch = None,
    train_error_estimate_batches = None,
    
    eval_data_pkl="test.pkl",
    eval_param_mins=torch.tensor([1.5, 0, -3.14]),
    eval_param_maxes=torch.tensor([7, np.deg2rad(65), 3.14]),
    coverage_calc_grid_num_points=10_000,
    eval_limit_count=20,
    confidence_levels=[0.9],
    
    calibration_data_pkl="calibration.pkl",
    calibration_batch_limit_count=None,
    calibration_num_posterior_samples=1000
)

DEBUG = ExperimentConfig(
    seed=0,
    gpu_index=1,
    source_data_config=DC_TRAIN_IS_TEST,
    detector_layout=torch.ones(2000, 2000),
    max_shift_radius=800,
    downsample_factor=20,
    min_features_threshold=10,
    time_averages=[True, False, False],
    
    model_constructor= lambda d, num_channels, fgs: nde_models.JointCRNF(
        device=d,
        context_model=nde_models.ContextModel(
            device=d,
            num_channels=num_channels,
            final_grid_shape=fgs,
            context_size=256,
            kernel_size=13
        ),
        num_flows=4
    ),
    model_save_path="debug.pt",
    
    train_data_pkl="train.pkl",
    val_data_pkl="val.pkl",
    train_param_mins=torch.tensor([1.5, np.deg2rad(42), np.deg2rad(-60)], dtype=torch.float32),
    train_param_maxes=torch.tensor([5.5, np.deg2rad(65), np.deg2rad(60)], dtype=torch.float32),
    train_batch_size=2048,
    learning_rate=1e-3,
    val_epoch_cooldown=10,
    max_epochs=2,
    train_limit_batches_per_epoch = None,
    train_error_estimate_batches = None,
    
    eval_data_pkl="test.pkl",
    eval_param_mins=torch.tensor([1.5, 0, -3.14]),
    eval_param_maxes=torch.tensor([7, np.deg2rad(65), 3.14]),
    coverage_calc_grid_num_points=10_000,
    eval_limit_count=20,
    confidence_levels=[0.9],
    
    calibration_data_pkl="calibration.pkl",
    calibration_batch_limit_count=None,
    calibration_num_posterior_samples=1000
)

CALIBRATE_ON_TRAIN = copy.deepcopy(DEFAULT_CONFIG)
CALIBRATE_ON_TRAIN.calibration_data_pkl = "train.pkl"
CALIBRATE_ON_TRAIN.calibration_batch_limit_count = 4

CALIBRATE_ON_BOTH_CAL_AND_TRAIN = copy.deepcopy(DEFAULT_CONFIG)
CALIBRATE_ON_BOTH_CAL_AND_TRAIN.additional_calibration_data_pkl = "train.pkl"
CALIBRATE_ON_BOTH_CAL_AND_TRAIN.calibration_batch_limit_count = 4

ENERGY_ONLY = copy.deepcopy(DEFAULT_CONFIG)
ENERGY_ONLY.source_data_config = DC_TRAIN_IS_TEST_ENERGY_ONLY
ENERGY_ONLY.model_save_path = "energy_only_joint.pt"
ENERGY_ONLY.calibration_batch_limit_count = None
ENERGY_ONLY.train_limit_batches_per_epoch = 60
ENERGY_ONLY.use_existing_trained_model = False
ENERGY_ONLY.train_error_estimate_batches = 60

ENERGY_ONLY_NO_SHIFT = copy.deepcopy(DEFAULT_CONFIG)
ENERGY_ONLY_NO_SHIFT.source_data_config = DC_TRAIN_IS_TEST_ENERGY_ONLY
ENERGY_ONLY_NO_SHIFT.model_save_path = "energy_only_no_shift_joint.pt"
ENERGY_ONLY_NO_SHIFT.calibration_batch_limit_count = None
ENERGY_ONLY_NO_SHIFT.train_limit_batches_per_epoch = 30
ENERGY_ONLY_NO_SHIFT.use_existing_trained_model = False
ENERGY_ONLY_NO_SHIFT.max_shift_radius = 0

ENERGY_ONLY_NO_SHIFT_BIGGER = copy.deepcopy(DEFAULT_CONFIG)
ENERGY_ONLY_NO_SHIFT_BIGGER.source_data_config = DC_TRAIN_IS_TEST_ENERGY_ONLY
ENERGY_ONLY_NO_SHIFT_BIGGER.model_save_path = "energy_only_no_shift_joint_bigger.pt"
ENERGY_ONLY_NO_SHIFT_BIGGER.train_limit_batches_per_epoch = 30
ENERGY_ONLY_NO_SHIFT_BIGGER.use_existing_trained_model = False
ENERGY_ONLY_NO_SHIFT_BIGGER.max_shift_radius = 0
ENERGY_ONLY_NO_SHIFT_BIGGER.model_constructor = lambda d, num_channels, fgs: nde_models.JointCRNF(
    device=d,
    context_model=nde_models.BiggerContextModel(
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    ),
    num_flows=4
)
ENERGY_ONLY_NO_SHIFT_BIGGER.learning_rate = 1e-4

ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER = copy.deepcopy(DEFAULT_CONFIG)
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.source_data_config = DC_PRIOR_SHIFT_ENERGY_ZENITH
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.model_save_path = "energy_zenith_no_shift_joint_bigger.pt"
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.train_limit_batches_per_epoch = 30
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.max_shift_radius = 0
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.model_constructor = lambda d, num_channels, fgs: nde_models.JointCRNF(
    device=d,
    context_model=nde_models.BiggerContextModel(
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    ),
    num_flows=4
)
ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER.learning_rate = 1e-3

TRIMMED_PRIORS = copy.deepcopy(DEFAULT_CONFIG)
TRIMMED_PRIORS.source_data_config = DC_TRIMMED_PRIORS
TRIMMED_PRIORS.model_save_path = "trimmed_priors.pt"
TRIMMED_PRIORS.max_shift_radius = 0
TRIMMED_PRIORS.train_limit_batches_per_epoch = 30
TRIMMED_PRIORS.train_param_mins = torch.tensor([3, np.deg2rad(35), np.deg2rad(-60)], dtype=torch.float32)
TRIMMED_PRIORS.train_param_maxes = torch.tensor([5.5, np.deg2rad(65), np.deg2rad(60)], dtype=torch.float32)
TRIMMED_PRIORS.eval_param_mins = torch.tensor([3, 0.2, -3.14])
TRIMMED_PRIORS.eval_param_maxes = torch.tensor([6, np.deg2rad(65), 3.14])
TRIMMED_PRIORS.train_sample_ratio = 0.35
TRIMMED_PRIORS.val_sample_ratio = 0.5
TRIMMED_PRIORS.model_constructor = lambda d, num_channels, fgs: nde_models.JointCRNF(
    device=d,
    context_model=nde_models.BiggerContextModel(
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    ),
    num_flows=4
)
TRIMMED_PRIORS.train_error_estimate_batches = 10

FULL_PRIORS = copy.deepcopy(DEFAULT_CONFIG)
FULL_PRIORS.source_data_config = DC_FULL_PRIORS
FULL_PRIORS.model_save_path = "full_priors.pt"
FULL_PRIORS.max_shift_radius = 800
FULL_PRIORS.train_limit_batches_per_epoch = 30
FULL_PRIORS.train_param_mins = torch.tensor([3, np.deg2rad(35), np.deg2rad(-60)], dtype=torch.float32)
FULL_PRIORS.train_param_maxes = torch.tensor([5.5, np.deg2rad(65), np.deg2rad(60)], dtype=torch.float32)
FULL_PRIORS.eval_param_mins = torch.tensor([3, 0, -3.14])
FULL_PRIORS.eval_param_maxes = torch.tensor([6, np.deg2rad(65), 3.14])
FULL_PRIORS.train_sample_ratio = 0.35
FULL_PRIORS.val_sample_ratio = 0.5
FULL_PRIORS.model_constructor = lambda d, num_channels, fgs: nde_models.JointCRNF(
    device=d,
    context_model=nde_models.BiggerContextModel(
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    ),
    num_flows=4
)
FULL_PRIORS.train_error_estimate_batches = 10
FULL_PRIORS.cal_loader_repeats = 5

FULL_PRIORS_RESTRICT_AZIMUTH = copy.deepcopy(FULL_PRIORS)
FULL_PRIORS_RESTRICT_AZIMUTH.restrict_azimuth_to_pm90deg = True
FULL_PRIORS_RESTRICT_AZIMUTH.train_differential_flux = crs.differential_crab_flux
FULL_PRIORS_RESTRICT_AZIMUTH.model_save_path = "full_priors_pm90.pt"
FULL_PRIORS_RESTRICT_AZIMUTH.cal_loader_repeats = 10
FULL_PRIORS_RESTRICT_AZIMUTH.source_data_config = DC_FLEXIBLE_PRIORS
FULL_PRIORS_RESTRICT_AZIMUTH.eval_param_mins=torch.tensor([3, 0, -np.pi/2])
FULL_PRIORS_RESTRICT_AZIMUTH.eval_param_maxes=torch.tensor([6, np.deg2rad(65), np.pi/2])

FULL_PRIORS_RESTRICT_AZIMUTH_LIMIT_CAL = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH)
FULL_PRIORS_RESTRICT_AZIMUTH_LIMIT_CAL.cal_loader_repeats = 3
FULL_PRIORS_RESTRICT_AZIMUTH_LIMIT_CAL.cal_limit_zenith_to_train = True

FULL_PRIORS_RESTRICT_AZIMUTH_NO_SHIFT = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH)
FULL_PRIORS_RESTRICT_AZIMUTH_NO_SHIFT.max_shift_radius = 0
FULL_PRIORS_RESTRICT_AZIMUTH_NO_SHIFT.cal_loader_repeats = 1
FULL_PRIORS_RESTRICT_AZIMUTH_NO_SHIFT.model_save_path = "full_priors_pm90_no_shift.pt"

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST.model_save_path = "full_priors_pm90_uniform_test.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST.source_data_config = DC_FLEXIBLE_PRIORS_UNIFORM_TEST
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST.train_sample_ratio = 0.23
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST.val_sample_ratio = 0.6

crab_trajectory = crs.get_source_trajectory("crab", -15)
crab_min_zenith = np.deg2rad(90 - np.array(crab_trajectory.alt).max())

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL.cal_loader_repeats = 1
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL.cal_limit_zenith_to_train = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL.calibration_num_posterior_samples = 100_000
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL.eval_param_mins = torch.tensor([3, crab_min_zenith, -np.pi/2])


FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.model_save_path = "full_priors_pm90_no_shift_uniform_test.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.source_data_config = DC_FLEXIBLE_PRIORS_UNIFORM_TEST
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.train_sample_ratio = 0.23
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.val_sample_ratio = 0.6
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.max_shift_radius = 0
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.cal_loader_repeats = 1
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.cal_limit_zenith_to_train = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.calibration_num_posterior_samples = 100_000 # {'n_estimators': 500, 'max_depth': 15}
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT.eval_param_mins = torch.tensor([3, crab_min_zenith, -np.pi/2])

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CORRECTED_CODE = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CORRECTED_CODE.model_save_path = "full_priors_pm90_no_shift_uniform_test_corrected_code.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CORRECTED_CODE.gpu_index = 0

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_POSTERIOR_TS = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_POSTERIOR_TS.use_posterior = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_POSTERIOR_TS.gpu_index = 1

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.model_save_path = "full_priors_pm90_no_shift_uniform_test_custom_prior.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.train_higher_energy_prior_config = {
    "num_bins": 30,
    "target_size": 155_000
}

FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR)
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.no_azimuth = True
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.model_save_path = "full_priors_pm90_no_shift_uniform_test_custom_prior_no_azimuth.pt"
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.train_higher_energy_prior_config = {
    "num_bins": 30,
    "target_size": 100_000
}
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.restrict_azimuth_to_pm90deg = False
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.eval_param_mins = torch.tensor([3, crab_min_zenith], dtype=torch.float32)
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.eval_param_maxes = torch.tensor([6, np.deg2rad(65)], dtype=torch.float32)
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.train_param_mins = torch.tensor([3, crab_min_zenith], dtype=torch.float32)
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.train_param_maxes = torch.tensor([6, np.deg2rad(65)], dtype=torch.float32)

FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR.model_constructor = lambda d, num_channels, fgs: nde_models.JointCRNF(
    device=d,
    context_model=nde_models.ContextModel(
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    ),
    num_flows=3,
    no_azimuth=True
)

FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR_SMALLER = copy.deepcopy(FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR)
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR_SMALLER.train_higher_energy_prior_config = {
    "num_bins": 30,
    "target_size": 50_000
}
FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR_SMALLER.model_save_path = "full_priors_pm90_no_shift_uniform_test_custom_prior_no_azimuth_smaller.pt"

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.use_sbi = True 
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.model_constructor = lambda d, num_channels, fgs: nde_models.WeightedNPSE(
    score_estimator=lambda x, y: VPScoreEstimator(
        net = nde_models.ScoreModel(
            device=d,
            num_channels=num_channels,
            final_grid_shape=fgs,
            kernel_size=13,
            t_embedding_dim=1,
            param_dim=3
        ),
        input_shape=torch.Size([3]),
        condition_shape=torch.Size([3, 100, 100])
    ),
    device=f"cuda:{d.index}"
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.model_save_path = "full_priors_pm90_no_shift_uniform_test_npse.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.gpu_index = 1
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.train_sample_ratio = 0.23 * 0.5
# FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.calibration_batch_limit_count = 100
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE.calibration_num_posterior_samples = 10_000

from sbi.neural_nets.net_builders.flowmatching_nets import build_mlp_flowmatcher

def fmpe_constructor(d, num_channels, fgs):
    context_model = nde_models.ContextModel(
        # device=torch.device("cuda:1"),
        device=d,
        num_channels=num_channels,
        final_grid_shape=fgs,
        context_size=256,
        kernel_size=13
    )
    context_model.eval()
    return nde_models.WeightedFMPE(
        prior=None,
        density_estimator=lambda theta, x: build_mlp_flowmatcher(
            theta,
            x,
            embedding_net=context_model,
            z_score_x=None,
            z_score_y=None
        ),
        device=d
    )

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.use_sbi = True 
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.model_constructor = fmpe_constructor
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.gpu_index = 1
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.train_sample_ratio = 0.23 * 0.5
# FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.calibration_batch_limit_count = 100
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE.calibration_num_posterior_samples = 10_000

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_POSTERIOR_SAMPLES = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_POSTERIOR_SAMPLES.calibration_num_posterior_samples = 100_000

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA.train_sample_ratio = 0.35
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_more_train_data.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA.gpu_index = 2

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA_POSTERIOR_TS = copy.deepcopy(FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA_POSTERIOR_TS.use_posterior = True

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA_POSTERIOR_TS
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS.train_sample_ratio = 1.0
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_posterior_ts.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS.train_ds_on_cpu = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS.gpu_index = 0

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS.use_posterior = False

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS.no_train_weights = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS.max_sbi_energy = 5.0
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_posterior_ts_no_weights.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS.eval_param_maxes = torch.tensor([5, np.deg2rad(65), np.pi/2], dtype=torch.float32)

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E.max_sbi_energy = 6.0
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_posterior_ts_no_weights_6E.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E.eval_param_maxes = torch.tensor([6, np.deg2rad(65), np.pi/2], dtype=torch.float32)

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_NO_WEIGHTS_6E = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_NO_WEIGHTS_6E.use_posterior = False

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV.eval_param_maxes = torch.tensor([4, np.deg2rad(65), np.pi/2], dtype=torch.float32)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_waldo_ts_4gev.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV.source_data_config = DC_FLEXIBLE_PRIORS_UNIFORM_TEST_4GEV

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV.use_posterior = True
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV.gpu_index = 0   

crab_trajectory_hawc = crs.get_source_trajectory("crab", 19)
crab_min_zenith_hawc = np.deg2rad(90 - np.array(crab_trajectory_hawc.alt).max())

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC.train_observer_latitude = 19
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC.eval_param_mins = torch.tensor([3, crab_min_zenith_hawc, -np.pi/2], dtype=torch.float32)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_waldo_ts_4gev_hawc.pt"

FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC_SUBSAMPLE = copy.deepcopy(
    FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC
)
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC_SUBSAMPLE.model_save_path = "full_priors_pm90_no_shift_uniform_test_fmpe_all_train_data_waldo_ts_4gev_hawc_subsample.pt"
FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC_SUBSAMPLE.use_subsampling = True



config_dict = {
    "default": DEFAULT_CONFIG,
    "calibrate_on_train": CALIBRATE_ON_TRAIN,
    "calibrate_on_mix": CALIBRATE_ON_BOTH_CAL_AND_TRAIN,
    "energy_only": ENERGY_ONLY,
    "energy_only_no_shift": ENERGY_ONLY_NO_SHIFT,
    "energy_only_no_shift_bigger": ENERGY_ONLY_NO_SHIFT_BIGGER,
    "energy_zenith_no_shift_bigger": ENERGY_ZENITH_NO_SHOWERSHIFT_BIGGER,
    "trimmed_priors": TRIMMED_PRIORS,
    "full_priors": FULL_PRIORS, #QR size 462268
    "full_priors_restrict_azimuth": FULL_PRIORS_RESTRICT_AZIMUTH, 
    "full_priors_restrict_azimuth_limit_cal": FULL_PRIORS_RESTRICT_AZIMUTH_LIMIT_CAL,
    "full_priors_restrict_azimuth_uniform_test": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST,
    "full_priors_restrict_azimuth_uniform_test_limit_cal": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL,
    "full_priors_restrict_azimuth_no_shift": FULL_PRIORS_RESTRICT_AZIMUTH_NO_SHIFT,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_corrected_code": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CORRECTED_CODE,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_posterior_ts": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_POSTERIOR_TS,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_custom_prior": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_npse": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_NPSE,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_more_posterior_samples": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_POSTERIOR_SAMPLES,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_more_train_data": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_more_train_data_posterior_ts": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_MORE_TRAIN_DATA_POSTERIOR_TS,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_waldo_ts": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_waldo_ts_4gev": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_4GEV,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts_4gev": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts_4gev_hawc": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC, # this is the one
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts_4gev_hawc_subsample": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_4GEV_HAWC_SUBSAMPLE,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts_no_weights": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_posterior_ts_no_weights_6E": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_POSTERIOR_TS_NO_WEIGHTS_6E,
    "full_priors_restrict_azimuth_uniform_test_limit_cal_no_shift_fmpe_all_train_data_waldo_ts_no_weights_6E": FULL_PRIORS_RESTRICT_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_FMPE_ALL_TRAIN_DATA_WALDO_TS_NO_WEIGHTS_6E,
    "full_priors_no_azimuth_uniform_test_limit_cal_no_shift_custom_prior": FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR,
    "full_priors_no_azimuth_uniform_test_limit_cal_no_shift_custom_prior_smaller": FULL_PRIORS_NO_AZIMUTH_UNIFORM_TEST_LIMIT_CAL_NO_SHIFT_CUSTOM_PRIOR_SMALLER,
    "debug": DEBUG
}