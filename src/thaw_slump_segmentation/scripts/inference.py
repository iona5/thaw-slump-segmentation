#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Inference Script
"""

import argparse
from pathlib import Path

import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from ..models import create_model
from ..utils.plot_info import flatui_cmap
from ..utils import init_logging, get_logger, log_run
from ..data_pre_processing import gdal

from ..scripts.setup_raw_data import preprocess_directory
from ..data_loading import DataSources

import yaml

cmap_prob = flatui_cmap('Midnight Blue', 'Alizarin')
cmap_dem = flatui_cmap('Alizarin', 'Clouds', 'Peter River')
cmap_slope = flatui_cmap('Clouds', 'Midnight Blue')
cmap_ndvi = 'RdYlGn'

FIGSIZE_MAX = 20

parser = argparse.ArgumentParser(description='Inference Script', 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gdal_bin", default='', help="Path to gdal binaries")
parser.add_argument("--gdal_path", default='', help="Path to gdal scripts")
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
parser.add_argument("--ckpt", default='latest', type=str, help="Checkpoint to use")
parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
parser.add_argument("--inference_dir", default='inference', type=Path, help="Main inference directory")
parser.add_argument("-n", "--name", default=None, type=str, help="Name of inference run, data will be stored in subdirectory")
parser.add_argument("-m", "--margin_size", default=256, type=int, help="Size of patch overlap")
parser.add_argument("-p", "--patch_size", default=1024, type=int, help="Size of patches")
parser.add_argument("model_path", type=str, help="path to model, use the model base path")
parser.add_argument("tile_to_predict", type=str, help="path to model", nargs='+')

args = parser.parse_args()
gdal.initialize(args)


def predict(model, imagery, device='cpu'):
    prediction = torch.zeros(1, *imagery.shape[2:])
    weights = torch.zeros(1, *imagery.shape[2:])

    PS = args.patch_size
    MARGIN = args.margin_size

    margin_ramp = torch.cat([
        torch.linspace(0, 1, MARGIN),
        torch.ones(PS - 2 * MARGIN),
        torch.linspace(1, 0, MARGIN),
    ])

    soft_margin = margin_ramp.reshape(1, 1, PS) * \
                  margin_ramp.reshape(1, PS, 1)

    for y in np.arange(0, imagery.shape[2], (PS - MARGIN)):
        for x in np.arange(0, imagery.shape[3], (PS - MARGIN)):
            if y + PS > imagery.shape[2]:
                y = imagery.shape[2] - PS
            if x + PS > imagery.shape[3]:
                x = imagery.shape[3] - PS
            patch_imagery = imagery[:, :, y:y + PS, x:x + PS]
            patch_pred = torch.sigmoid(model(patch_imagery.to(device))[0].cpu())

            # Essentially premultiplied alpha blending
            prediction[:, y:y + PS, x:x + PS] += patch_pred * soft_margin
            weights[:, y:y + PS, x:x + PS] += soft_margin

    # Avoid division by zero
    weights = torch.where(weights == 0, torch.ones_like(weights), weights)
    return prediction / weights


def flush_rio(filepath):
    """For some reason, rasterio doesn't actually finish writing
    a file after finishing a `with rio.open(...) as ...:` block
    Trying to open the file for reading seems to force a flush"""

    with rio.open(filepath) as f:
        pass


def do_inference(tilename, sources, model, dev, logger, args=None, log_path=None):
    tile_logger = get_logger(f'inference.{tilename}')
    # ===== PREPARE THE DATA =====
    DATA_ROOT = args.data_dir
    INFERENCE_ROOT = args.inference_dir
    data_directory = DATA_ROOT / 'tiles' / tilename
    if not data_directory.exists():
        logger.info(f'Preprocessing directory {tilename}')
        raw_directory = DATA_ROOT / 'input' / tilename
        if not raw_directory.exists():
            logger.error(f"Couldn't find tile '{tilename}' in {DATA_ROOT}/tiles or {DATA_ROOT}/input. Skipping this tile")
            return
        preprocess_directory(raw_directory, args, log_path, label_required=False)
        # After this, data_directory should contain all the stuff that we need.
    
    if args.name:
        output_directory = INFERENCE_ROOT / args.name / tilename
    else:
        output_directory = INFERENCE_ROOT / tilename
    
    output_directory.mkdir(exist_ok=True, parents=True)

    planet_imagery_path = next(data_directory.glob('*_SR.tif'))

    data = []
    for source in sources:
        tile_logger.debug(f'loading {source.name}')
        if source.name == 'planet':
            tif_path = planet_imagery_path
        else:
            tif_path = data_directory / f'{source.name}.tif'

        data_part = rio.open(tif_path).read().astype(np.float32)

        if source.name == 'tcvis':
            data_part = data_part[:3]
        data_part = np.nan_to_num(data_part, nan=0.0)

        data_part = data_part / np.array(source.normalization_factors, dtype=np.float32).reshape(-1, 1, 1)
        data.append(data_part)

    def make_img(filename, source, colorbar=False, mask=None, **kwargs):
        idx = sources.index(source)
        h, w = data[idx].shape[1:]
        if mask is None:
            mask = np.zeros((h, w), dtype=np.bool)
        if h > w:
            figsize = (FIGSIZE_MAX * w / h, FIGSIZE_MAX)
        else:
            figsize = (FIGSIZE_MAX, FIGSIZE_MAX * h / w)
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        if source.channels >= 3:
            rgb = np.stack([np.ma.masked_where(mask, data[idx][i])[::10, ::10] for i in range(3)], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, aspect='equal')
        elif source.channels == 1:
            image = ax.imshow(np.ma.masked_where(mask, data[idx][0]), aspect='equal', **kwargs)
            if colorbar:
                plt.colorbar(image)
        plt.savefig(output_directory / filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_results(image, outfile):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        mappable = ax.imshow(image, vmin=0, vmax=1, cmap=cmap_prob, aspect='equal')
        plt.colorbar(mappable)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
        plt.close()


    full_data = np.concatenate(data, axis=0)
    nodata = np.all(full_data == 0, axis=0, keepdims=True)
    full_data = torch.from_numpy(full_data)
    full_data = full_data.unsqueeze(0)  # Pretend this is a batch of size 1

    res = predict(model, full_data, dev).numpy()
    del full_data

    res[nodata] = np.nan
    binarized = np.ones_like(res, dtype=np.uint8) * 255
    binarized[~nodata] = (res[~nodata] > 0.5).astype(np.uint8)

    # define output file paths
    out_path_proba = output_directory / 'pred_probability.tif'
    out_path_label = output_directory / 'pred_binarized.tif'
    out_path_pre_poly = output_directory / 'pred_binarized_tmp.tif'
    out_path_shp = output_directory / 'pred_binarized.shp'
    out_path_gpkg = output_directory / 'pred_binarized.gpkg'

    # Get the input profile
    with rio.open(planet_imagery_path) as input_raster:
        profile = input_raster.profile
        profile.update(
            dtype=rio.float32,
            count=1,
            compress='lzw',
            driver='COG',
            #tiled=True
        )

    with rio.open(out_path_proba, 'w', **profile) as output_raster:
        output_raster.write(res.astype(np.float32))
    flush_rio(out_path_proba)

    profile.update(
        dtype=rio.uint8,
        nodata=255
    )
    with rio.open(out_path_label, 'w', **profile) as output_raster:
        output_raster.write(binarized)
    flush_rio(out_path_label)

    with rio.open(out_path_pre_poly, 'w', **profile) as output_raster:
        output_raster.write((binarized == 1).astype(np.uint8))
    flush_rio(out_path_pre_poly)

    # create vectors
    log_run(f'{gdal.polygonize} {out_path_pre_poly} -q -mask {out_path_pre_poly} -f "ESRI Shapefile" {out_path_shp}', tile_logger)
    log_run(f'{gdal.polygonize} {out_path_pre_poly} -q -mask {out_path_pre_poly} -f "GPKG" {out_path_gpkg}', tile_logger)
    #log_run(f'python {gdal.polygonize} {out_path_pre_poly} -q -mask {out_path_pre_poly} -f "ESRI Shapefile" {out_path_shp}', tile_logger)
    out_path_pre_poly.unlink()

    h, w = res.shape[1:]
    if h > w:
        figsize = (FIGSIZE_MAX * w / h, FIGSIZE_MAX)
    else:
        figsize = (FIGSIZE_MAX, FIGSIZE_MAX * h / w)

    for src in sources:
        kwargs = dict()
        if src.name == 'ndvi':
            kwargs = dict(colorbar=True, cmap=cmap_ndvi, vmin=0, vmax=1)
        elif src.name == 'relative_elevation':
            kwargs = dict(colorbar=True, cmap=cmap_dem, vmin=0, vmax=1)
        elif src.name == 'slope':
            kwargs = dict(colorbar=True, cmap=cmap_slope, vmin=0, vmax=0.5)
        make_img(f'{src.name}.jpg', src, mask=nodata[0],**kwargs)

    outpath = output_directory / 'pred_probability.jpg'
    plot_results(np.ma.masked_where(nodata[0], res[0]), outpath)

    outpath = output_directory / 'pred_binarized.jpg'
    plot_results(np.ma.masked_where(nodata[0], binarized[0]), outpath)

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.log_dir) / f'inference-{timestamp}.log'
    if not Path(args.log_dir).exists():
	    os.mkdir(Path(args.log_dir))
    init_logging(log_path)
    logger = get_logger('inference')

    # ===== LOAD THE MODEL =====
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    logger.info(f'Running on {dev} device')

    if not args.model_path:
        last_modified = 0
        last_modeldir = None

        for config_file in Path(args.log_dir).glob('*/config.yml'):
            modified = config_file.stat().st_mtime
            if modified > last_modified:
                last_modified = modified
                last_modeldir = config_file.parent
        args.model_path = last_modeldir

    model_dir = Path(args.model_path)
    model_name = model_dir.name
    config = yaml.load((model_dir / 'config.yml').open(), Loader=yaml.SafeLoader)

    m = config['model']
    #print(m['architecture'],m['encoder'], m['input_channels'])
    model = create_model(
        arch=m['architecture'],
        encoder_name=m['encoder'],
        encoder_weights=None if m['encoder_weights'] == 'random' else m['encoder_weights'],
        classes=1,
        in_channels=m['input_channels']
    )

    if args.ckpt == 'latest':
        ckpt_nums = [int(ckpt.stem) for ckpt in model_dir.glob('checkpoints/*.pt')]
        last_ckpt = max(ckpt_nums)
    else:
        last_ckpt = int(args.ckpt)
    ckpt = model_dir / 'checkpoints' / f'{last_ckpt:02d}.pt'
    logger.info(f"Loading checkpoint {ckpt}")
    
    # Parallelized Model needs to be declared before loading
    try:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    
    model = model.to(dev)

    sources = DataSources(config['data_sources'])

    torch.set_grad_enabled(False)

    for tilename in tqdm(args.tile_to_predict):
        do_inference(tilename, sources, model, dev, logger, args, log_path)

if __name__ == "__main__":
  main()
