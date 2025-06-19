import numpy as np
import torch
import configs
import pickle as pkl
import tiled_events as te
import posteriors as pos
import io
from tqdm import tqdm


def angle_line(azimuth, zenith):
    if zenith == 0:
        return (0, 0)
  
    return np.array([np.sin(-azimuth), np.cos(azimuth)])


def scale_param_grid(
    param_grid: torch.Tensor,
    train_param_mins: torch.Tensor,
    train_param_maxes: torch.Tensor
):
    return (param_grid - train_param_mins)/(train_param_maxes - train_param_mins)

def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
def load_config_data(
    config_name,
    results_dir: str,
    get_model: bool = False,
    get_test_ds: bool = False,
    get_qr_data: bool = False,
    get_qr: bool = False,
    manual_device: torch.device = None
):

    config = configs.config_dict[config_name]
    out_dir = results_dir + f"{config_name}_{config.seed}/"
    
    if get_model:
        if manual_device is None:
            device = torch.device(f"cuda:{config.gpu_index}")
        else:
            print("OVERWRITING DEFAULT DEVICE")
            device = manual_device
            
        model = config.model_constructor(
            device,
            len(config.time_averages), 
            torch.tensor(config.detector_layout.shape)/config.downsample_factor
        )
        if not config.use_sbi:
            model.load_state_dict(torch.load(results_dir + config.model_save_path, map_location=device))
        else:
        # open single event
            with open("single_event.pkl", 'rb') as file:
                event = pkl.load(file)
            model._neural_net = model._build_neural_net(event['params'].to(device)[None], event['features'].to(device)[None])
            model._neural_net.load_state_dict(torch.load(results_dir + config.model_save_path, map_location=device))
            model._neural_net.eval()
            model._neural_net.to(device)
    else:
        model = None

    if get_model:
        with open(out_dir + "train_feature_maxes.pkl", 'rb') as file:
            train_feature_maxes = CPU_Unpickler(file).load()
        
    if get_test_ds:
        with open(f"{config.source_data_config.splits_dir}test.pkl", "rb") as file:
            event_list = pkl.load(file)
            event_list = [e for e in tqdm(event_list, desc="Zenith Cut") if e.zenith >= config.eval_param_mins[1]]
            test_ds = te.CRImageDataset(
                events=event_list,
                downsample_factor=config.downsample_factor,
                detector_layout=config.detector_layout,
                max_shift_radius=config.max_shift_radius,
                min_nonzero_features=config.min_features_threshold,
                sample_ratio=1.0,
                restrict_to_pm90=config.restrict_azimuth_to_pm90deg,
                device=torch.device("cpu"),
                no_azimuth=config.no_azimuth
            )
        test_ds.set_feature_scale(train_feature_maxes)
        test_ds.set_param_weighter(-1)
        test_ds.max_shift_radius = config.max_shift_radius
        test_ds.output_manifest(out_dir + "test_manifest.pkl")

    else:
        test_ds = None

    if get_qr:
        qr = pos.lf2i_qr(
            config.confidence_levels[0],
            None,
            model,
            None,
            config.train_param_mins,
            config.train_param_maxes,
            out_dir,
            num_posterior_samples=config.calibration_num_posterior_samples,
            cal_loader_repeats=config.cal_loader_repeats
        )
    else:
        qr = None

    with open(f"{config.source_data_config.filtered_manifest_filename}", 'rb') as file:
        filtered_manifest = pkl.load(file)
        
    with open("vsi_all_manifest.pkl", 'rb') as file:
        unfiltered_manifest = pkl.load(file)
        

    if config.restrict_azimuth_to_pm90deg:
        filtered_manifest["azimuth"] = ((filtered_manifest["azimuth"] + np.pi/2) % np.pi) - np.pi/2
        unfiltered_manifest["azimuth"] = ((unfiltered_manifest["azimuth"] + np.pi/2) % np.pi) - np.pi/2

    if get_qr_data:
        with open(out_dir + "qr_data.pkl", 'rb') as file:
            qr_data = pkl.load(file)
    else:
        qr_data = None
        
    return config, out_dir, model, test_ds, qr, qr_data, filtered_manifest, unfiltered_manifest