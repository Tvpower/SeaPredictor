import rasterio, numpy as np
from PIL import Image
with rasterio.open('data/data/raw/MARIDA/patches/S2_18-9-20_16PCC/S2_18-9-20_16PCC_39.tif') as src:
    # MARIDA bands: 1=B01, 2=B02, 3=B03, 4=B04, 5=B05, 6=B06, 7=B07,
    #               8=B08, 9=B8A, 10=B11, 11=B12
    b6, b8, b11 = src.read([6, 8, 10]).astype(np.float32)
# Floating Debris Index (Biermann et al. 2020)
fdi = b8 - (b6 + (b11 - b6) * (834 - 665) / (1610 - 665))
fdi_norm = np.clip((fdi - np.percentile(fdi, 5)) /
                   (np.percentile(fdi, 99) - np.percentile(fdi, 5) + 1e-6), 0, 1)
heat = (fdi_norm * 255).astype(np.uint8)
# Make a red heatmap on a black background
viz = np.zeros((*heat.shape, 3), dtype=np.uint8)
viz[..., 0] = heat            # red channel = FDI strength
Image.fromarray(viz).save('/tmp/fdi.png')