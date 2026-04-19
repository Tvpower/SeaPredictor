import numpy as np

EPS = 1e-10


def compute_indices(bands: np.ndarray) -> dict:
    """Input: (11, H, W). Returns dict of index_name -> (H, W) arrays."""
    B3 = bands[2]; B4 = bands[3]; B5 = bands[4]; B6 = bands[5]
    B7 = bands[6]; B8 = bands[7]; B8A = bands[8]; B11 = bands[9]; B12 = bands[10]

    ndvi = (B8 - B4) / (B8 + B4 + EPS)
    ndwi = (B3 - B8) / (B3 + B8 + EPS)
    fdi  = B8 - (B6 + (B11 - B6) * ((842 - 740) / (1610 - 740)))
    fai  = B8 - (B4 + (B12 - B4) * ((842 - 665) / (2190 - 665)))
    ndmi = (B8 - B11) / (B8 + B11 + EPS)
    bsi  = ((B11 + B4) - (B8 + B3)) / ((B11 + B4) + (B8 + B3) + EPS)
    nrd  = (B5 - B6) / (B5 + B6 + EPS)
    si   = np.cbrt((1 - B3) * (1 - B4) * (1 - B6))

    return {'NDVI': ndvi, 'NDWI': ndwi, 'FDI': fdi, 'FAI': fai,
            'NDMI': ndmi, 'BSI': bsi, 'NRD': nrd, 'SI': si}


def stack_indices(bands: np.ndarray) -> np.ndarray:
    """Returns (8, H, W) array of spectral indices."""
    indices = compute_indices(bands)
    return np.stack(list(indices.values()), axis=0).astype(np.float32)


def validate_indices(index_stack: np.ndarray) -> bool:
    has_nan = np.isnan(index_stack).any()
    has_inf = np.isinf(index_stack).any()
    if has_nan or has_inf:
        print(f"WARNING: indices contain NaN={has_nan}, Inf={has_inf}")
        return False
    return True
