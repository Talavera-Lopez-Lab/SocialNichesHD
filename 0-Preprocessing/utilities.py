import numpy as np
from functools import reduce


def spatial_subset(adata, shapes, shape_params, actions):
    coordinates = adata.obsm["spatial"]
    def create_mask(shape, shape_param):
        if shape == "rectangle":
            x_min, y_min = shape_param[0]
            x_max, y_max = shape_param[1]
            mask = (coordinates[:, 0] >= x_min) & \
                (coordinates[:, 0] <= x_max) & \
                (coordinates[:, 1] >= y_min) & \
                (coordinates[:, 1] <= y_max)
        elif shape == "circle":
            x_center, y_center = shape_param[0]
            radius = shape_param[1]
            mask = ((coordinates[:, 0] - x_center)**2 + (coordinates[:, 1] - y_center)**2) <= radius**2
        else:
            raise ValueError("Unsupported shape. Choose 'rectangle' or 'circle'.")
        return mask
    masks = [np.full(shape=(coordinates.shape[0],), fill_value=False, dtype=bool)] + \
    list(zip(map(create_mask, shapes, shape_params), actions))
    return adata[reduce(
        lambda a, b: a | b[0] if b[1] == "keep" else a & ~b[0], 
        masks
    )]