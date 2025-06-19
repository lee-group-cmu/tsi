import tiled_events as te 
import posteriors as pos
import torch


def test_sparse_to_dense():
    test_input = te.SparseStackedGridFeatures(
        grid_x_list=[torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])],
        grid_y_list=[torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])],
        grid_values_list=[torch.tensor([4, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])],
        grid_length=4,
        is_time_column=[False]
    )
    
    assert (te.sparse_to_dense_grid_features(
        test_input,
        torch.ones(4, 4),
        downsample_factor=2,
        shower_shift_x=0,
        shower_shift_y=0
    ) == torch.tensor([[[4.75, 0], [0, 0]]])).all()
    
    test_input = te.SparseStackedGridFeatures(
        grid_x_list=[torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])],
        grid_y_list=[torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])],
        grid_values_list=[torch.tensor([4, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])],
        grid_length=4,
        is_time_column=[True]
    )
    
    assert (te.sparse_to_dense_grid_features(
        test_input,
        torch.ones(4, 4),
        downsample_factor=2,
        shower_shift_x=0,
        shower_shift_y=0
    ) == torch.tensor([[[5, 0], [0, 0]]])).all()
    
def test_param_grid():
    assert pos.get_param_grid(
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 1, 1]),
        target_grid_size=1000
    ).shape[0] == 1000
    
    test_val = pos.get_param_grid(
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 1, 1]),
        target_grid_size=1000,
        fixed_energy=0.5
    )
    
    assert test_val.shape[0] <= 1000
    assert (test_val[:, 0] == 0.5).all()