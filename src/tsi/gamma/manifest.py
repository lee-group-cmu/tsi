import argparse
import raw_events as rev
import tiled_events as te
import pickle as pkl
import universal
import sys
import numpy as np
from configs import dc_config_dict
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSI Cosmic Ray Experiments")
    
    # Add the arguments
    parser.add_argument('--data_config', type=str, required=True, help='Which data config')
    parser.add_argument('--overwrite', action="store_true", help='Regenerate base manifest?')
    parser.add_argument('--metrics_only', action="store_true", help='Show manifest totals only?')
    
    # Parse the arguments
    args = parser.parse_args()
    config = dc_config_dict[args.data_config]
    
    rev.generate_unfiltered_manifest(
        config.manifest_filename,
        [
            f"{config.data_dir}big/",
            f"{config.data_dir}small/"
        ],
        overwrite=args.overwrite
    )
    
    config.manifest_generator(
        config.manifest_filename,
        config.filtered_manifest_filename
    )
    
    with open(config.filtered_manifest_filename, 'rb') as file:
        manifest_df = pkl.load(file)
        
    print(f"Train: {(manifest_df['split'] == 1).sum()}")
    print(f"Calibration: {(manifest_df['split'] == 2).sum()}")
    print(f"Test: {(manifest_df['split'] == 3).sum()}")
    
    if args.metrics_only:
        sys.exit(0)


    ratios = [
        config.train_sample_ratio,
        config.cal_sample_ratio,
        config.test_sample_ratio
    ]
    for split_id, split_name in enumerate(["train", "calibration", "test"]):
        main_data = list()
        val_data = list()
        holdout_data = list()
        
        split_df = manifest_df.loc[manifest_df["split"] == split_id + 1].sample(frac=ratios[split_id])
        n = len(split_df)
        val_counter = -1
        
        file_list = split_df["file"].unique()

        for fid, filename in enumerate(tqdm(file_list)):
            with open(filename, 'rb') as file:
                tiled_event_dict = pkl.load(file)
            for row in split_df.loc[split_df["file"] == filename, :].itertuples():
                event = tiled_event_dict['events'][row.event_id]
                event_features = event['features']
                val_counter += 1
                if event_features is None:
                    continue
                if sum([len(val['ti']) for _, val in  event_features['global_features'].items()]) == 0:
                    continue
                tiled_event = te.TiledEvent(
                    primary_type=universal.PrimaryParticleId.GAMMA,
                    log10_energy_gev=np.log10(event['energy']),
                    zenith=event['zenith'],
                    azimuth=event['azimuth'],
                    time_tail_trims=event_features['time_tail_trims'],
                    global_features=event_features['global_features'],
                    channel_features=event_features['channel_features'],
                    grid_length=event_features["grid_length"]
                )
                if split_name == "train" and val_counter < n * config.val_of_train_ratio:
                    val_data.append(tiled_event)
                elif split_name == "train" and val_counter < n * (config.val_of_train_ratio + config.holdout_of_train_ratio):
                    holdout_data.append(tiled_event)
                else:
                    main_data.append(tiled_event)

                
        with open(f"{config.splits_dir}{split_name}.pkl", "wb") as file:
            pkl.dump(main_data, file)
            
        if split_name == "train":
            with open(f"{config.splits_dir}val.pkl", "wb") as file:
                pkl.dump(val_data, file)
            with open(f"{config.splits_dir}holdout.pkl", "wb") as file:
                pkl.dump(holdout_data, file)
        
        print(f"{split_name} split: {len(main_data)} | {len(val_data)} | {len(holdout_data)}")