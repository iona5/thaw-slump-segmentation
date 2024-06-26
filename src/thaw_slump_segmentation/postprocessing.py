from typing import Union, List, Optional
from pathlib import Path
import geopandas as gpd
import pandas as pd
import shutil
import numpy as np
import rasterio
import os
from skimage.morphology import disk, binary_dilation, label, remove_small_objects
from skimage.morphology import disk, dilation, remove_small_objects

try:
    from cucim.skimage.morphology import disk as disk_gpu
    from cucim.skimage.morphology import binary_dilation as binary_dilation_gpu
    import cupy as cp
    CUCIM_AVAILABLE = True
    #print('cuCIM is available for GPU image operations')
except:
    CUCIM_AVAILABLE = False
    print('Using standard skimage CPU implementation')

def run_inference(df, model, processing_dir, inference_dir, model_dir=Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'), gpu=0, run=False, patch_size=1024, margin_size=256):
    if len(df) == 0:
        print('Empty dataframe')
    else:
        tiles = ' '.join(df.name.values)
        run_string = f"CUDA_VISIBLE_DEVICES='{gpu}' inference -n {model} --data_dir {processing_dir} --inference_dir {inference_dir}  --patch_size {patch_size} --margin_size {margin_size} {model_dir/model} {tiles}"
        print(run_string)
        if run:
            os.system(run_string)

def listdirs(rootdir):
    dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            #print(path)
            dirs.append(path)
    return dirs

def listdirs2(rootdir, depth=0):
    dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if depth == 1:
                for path2 in Path(path).iterdir():
                    if path2.is_dir():
                        dirs.append(path2)
            else:
                dirs.append(path)
    return dirs

def get_PS_products_type(name):
    if len(name.split('_')) == 3:
        return 'PSScene'
    elif len(name.split('_')) == 4:
        if len(name.split('_')[0]) == 8:
            return 'PSScene'
        else:
            return 'PSOrthoTile'
    else:
        None
        
def get_date_from_PSfilename(name):
    date = name.split('_')[2]
    return date
    

# TODO: create empty dataframe if no files found
def get_datasets(path, depth=0, preprocessed=False):
    dirs = listdirs2(path, depth=depth)
    df = pd.DataFrame(data=dirs, columns=['path'])
    df['name'] = df.apply(lambda x: x['path'].name, axis=1)
    df['preprocessed'] = preprocessed
    df['PS_product_type'] = df.apply(lambda x: get_PS_products_type(x['name']), axis=1)
    df['image_date'] = df.apply(lambda x: get_date_from_PSfilename(x['name']), axis=1)
    df['tile_id'] = df.apply(lambda x: x['name'].split('_')[1], axis=1)
    return df

def copy_unprocessed_files(row, processing_dir, quiet=True):
    """
    Copies unprocessed files from a source directory to a target directory.

    This function checks if a file exists in the target directory. If the file does not exist, it copies the file from the source to the target directory. If the file already exists in the target directory, it skips the copying process. 

    Parameters:
    row (dict): A dictionary containing file information. It should have a key 'path' which corresponds to the file's path.
    processing_dir (Path): The target directory where the files should be copied to. It should be a pathlib.Path object.
    quiet (bool, optional): If set to True, the function will not print any output during its execution. If set to False, the function will print the status of the file copying process. Default is True.

    Returns:
    None
    """
    inpath = row['path']
    outpath = processing_dir / 'input' / inpath.name

    if not outpath.exists():
        if not quiet:
            print (f'Start copying {inpath.name} to {outpath}')
        shutil.copytree(inpath, outpath)
    else:
        if not quiet:
            print(f'Skipped copying {inpath.name}')

def update_DEM(vrt_target_dir):
    """
    Function to update elevation vrts
    """
    os.system('./create_ArcticDEM.sh')
    shutil.copy('elevation.vrt', vrt_target_dir)
    shutil.copy('slope.vrt', vrt_target_dir)


def update_DEM2(dem_data_dir, vrt_target_dir):
    """
    Update elevation and slope virtual raster tiles (VRTs) from a directory of GeoTIFF tiles.

    Args:
        dem_data_dir (Path): Path to the directory containing the elevation and slope GeoTIFF tiles.
        vrt_target_dir (Path): Path to the directory where the updated VRTs will be saved.

    This function creates two VRTs, one for elevation and one for slope, from the GeoTIFF tiles
    in the `dem_data_dir` directory. The VRTs are saved in the `vrt_target_dir` directory with
    the filenames 'elevation.vrt' and 'slope.vrt', respectively.

    The function uses the `gdalbuildvrt` command-line utility to create the VRTs, and it sets
    the nodata value for both the input tiles and the output VRTs to 0.
    """
    # elevation
    flist = list((dem_data_dir / 'tiles_rel_el').glob('*.tif'))
    flist = [f'{f.absolute().as_posix()}\n' for f in flist]
    file_list_text = Path(vrt_target_dir) / 'flist_rel_el.txt'
    with open(file_list_text, 'w') as src:
        src.writelines(flist)
    vrt_target = Path(vrt_target_dir) / 'elevation.vrt'
    s = f'gdalbuildvrt -input_file_list {file_list_text} -srcnodata "0" -vrtnodata "0" {vrt_target}'
    os.system(s)
    os.remove(file_list_text)

    # slope
    flist = list((dem_data_dir / 'tiles_slope').glob('*.tif'))
    flist = [f'{f.absolute().as_posix()}\n' for f in flist]
    file_list_text = Path(vrt_target_dir) / 'flist_slope.txt'
    with open(file_list_text, 'w') as src:
        src.writelines(flist)
    vrt_target = Path(vrt_target_dir) / 'slope.vrt'
    s = f'gdalbuildvrt -input_file_list {file_list_text} -srcnodata "0" -vrtnodata "0" {vrt_target}'
    os.system(s)
    os.remove(file_list_text)
    

def get_processing_status(raw_data_dir, processing_dir, inference_dir, model, reduce_to_raw=False):
    # get raw tiles
    try:
        df_raw = get_datasets(raw_data_dir, depth=1)
    except:
        df_raw = get_datasets(raw_data_dir, depth=0)
    # get processed
    # TODO: make validation steps if files are alright 
    df_processed = get_datasets(processing_dir / 'tiles', depth=0, preprocessed=True)
    # check if all files are available
    df_processed = df_processed[df_processed.apply(lambda x: len(list(x['path'].glob('*')))>=5, axis=1)]

    # get all non preprocessed raw images
    diff = df_raw[~df_raw['name'].isin(df_processed['name'])]
    # TODO: issue here
    if reduce_to_raw == True:
        # TODO: check 
        df_merged = diff
    else:
        df_merged = pd.concat([df_processed, diff]).reset_index()


    products_list = [prod.name for prod in list((inference_dir / model).glob('*'))]
    df_merged['inference_finished'] = df_merged.apply(lambda x: x['name'] in (products_list), axis=1)
    
    return df_merged


def get_processing_status_ensemble(inference_dir, model_input_names=['RTS_v5_notcvis','RTS_v5_tcvis'], model_ensemble_name='RTS_v5_ensemble'):
    """
    Get processing status for a model ensemble and its individual models based on available data.

    This function examines the contents of specified directories within the 'inference_dir'
    to determine the processing status of a model ensemble and its constituent models.
    It constructs DataFrames indicating whether data is available for each model, and whether
    the processing has been completed for both the ensemble and individual models.

    Args:
        inference_dir (Path-like): Path to the directory containing inference data.
        model_input_names (list, optional): List of model input directory names.
            Default values are ['RTS_v5_notcvis', 'RTS_v5_tcvis'].
        model_ensemble_name (str, optional): Name of the model ensemble directory.
            Default value is 'RTS_v5_ensemble'.

    Returns:
        pandas.DataFrame: A DataFrame containing the processing status for each model
        and the ensemble. Columns include 'name', 'data_available', and 'process'.

    Example:
        >>> inference_dir = Path('/path/to/inference_data')
        >>> status_df = get_processing_status_ensemble(inference_dir)

    """
    dfs = []
    for model in model_input_names[:]:
        ds_names = [prod.name for prod in list((inference_dir / model).glob('*')) if prod.is_dir()]
        has_proba = [(inference_dir / model / f / 'pred_probability.tif').exists() for f in ds_names]
        df = pd.DataFrame(data=ds_names, columns=['name']).set_index('name')
        df['model_name'] = model
        df['has_proba'] = has_proba
        dfs.append(df)

    ds_names_ensemble = [prod.name for prod in list((inference_dir / model_ensemble_name).glob('*')) if prod.is_dir()]
    df_ensemble = pd.DataFrame(data=ds_names_ensemble, columns=['name']).set_index('name')
    df_ensemble['ensemble_name'] = model_ensemble_name
    dfs.append(df_ensemble)

    df_process = pd.concat(dfs, axis=1)

    df_process['data_available'] = ~df_process['model_name'].isna().any(axis=1)
    df_process['proba_available'] = df_process['has_proba'].all(axis=1)
    df_process['process'] = df_process['data_available'] & (df_process['ensemble_name'].isna() & df_process['proba_available'])

    return df_process[['process', 'data_available', 'proba_available']].reset_index(drop=False).rename(columns={'index':'name'})

    
def create_ensemble(inference_dir: Path, modelnames: List[str], ensemblename: str, image_id: str, binary_threshold: list=[0.3,0.4,0.5], delete_proba=True, delete_binary=True):
    """
    Calculate the mean of two model predictions and write the output to disk.
    
    Args:
    modelnames (List[str]): A list of two model names.
    ensemblename (str): The name of the ensemble model.
    image_id (str): The ID of the image.
    binary_threshold (float): The binary threshold value.
    
    Returns:
    None
    """
    try:
        # setup
        outpath = inference_dir / ensemblename / image_id / f'{image_id}_{ensemblename}_proba.tif'
        os.makedirs(outpath.parent, exist_ok=True)

        # calculate
        image1 = inference_dir / modelnames[0] / image_id / 'pred_probability.tif'
        image2 = inference_dir / modelnames[1] / image_id / 'pred_probability.tif'

        with rasterio.open(image1) as src1:
            with rasterio.open(image2) as src2:
                a1 = src1.read()
                a2 = src2.read()

            out_meta = src1.meta.copy()
            out_meta_binary = out_meta.copy()
            out_meta_binary['dtype'] = 'uint8'

        out = np.mean([a1, a2], axis=0)
        with rasterio.open(outpath, 'w', **out_meta) as target:
            target.write(out)


        # write binary raster
        for threshold in binary_threshold:
            thresh_str = str(threshold).replace('.','')
            outpath_class = Path(str(outpath).replace('proba', f'class_{thresh_str}'))
            outpath_shp = outpath_class.with_suffix('.gpkg')

            out_binary = (out >= threshold)

            with rasterio.open(outpath_class, 'w', **out_meta_binary, compress='deflate') as target:
                target.write(out_binary)

            # make vector
            s_polygonize = f'gdal_polygonize.py {outpath_class} -q -mask {outpath_class} -f "GPKG" {outpath_shp}'
            os.system(s_polygonize)
            if delete_binary:
                os.remove(outpath_class)

        # delete files
        if delete_proba:
            os.remove(outpath)
            
        return 0
    
    except:
        return 1
    
    
def create_ensemble_v2(inference_dir: Path, 
                        modelnames: List[str], 
                        ensemblename: str, 
                        image_id: str, 
                        binary_threshold: list=[0.5], 
                        border_size: int=10,
                        minimum_mapping_unit: int=32,
                        delete_binary: bool=True,
                        try_gpu: bool=True,
                        gpu: int=0):
    """
    Create an ensemble result from multiple model predictions, generate binary masks, and process the output.

    Parameters:
    ------------
    inference_dir : Path
        Path to the directory containing inference data.

    modelnames : List[str]
        List of model names to be used for creating the ensemble.

    ensemblename : str
        Name of the ensemble result.

    image_id : str
        Identifier for the image being processed.

    binary_threshold : list, optional
        List of binary threshold values, by default [0.5].

    border_size : int, optional
        Border size for edge masking and dilation, by default 10.

    minimum_mapping_unit : int, optional
        Minimum size of mapping units to retain, by default 32.

    delete_binary : bool, optional
        Whether to delete the binary file after processing, by default True.

    Returns:
    ------------
    None
    """
    
    def calculate_mean_image(images):
        ctr = 0
        list_data = []

        for image in images:
            with rasterio.open(image) as src:
                data = src.read()
                if ctr == 0:
                    out_meta = src.meta.copy()
                    out_meta_binary = out_meta.copy()
                    out_meta_binary['dtype'] = 'uint8'
            list_data.append(data)
            ctr += 1

        mean_image = np.mean(list_data, axis=0)
        return mean_image, out_meta_binary

    def dilate_data_mask(mask, size=10):
        selem = disk(size)
        return binary_dilation(mask, selem)
    
    def dilate_data_mask_gpu(mask, size=10):
        mask = cp.array(mask)
        selem = disk_gpu(size)
        return cp.asnumpy(binary_dilation_gpu(mask, selem))

    def mask_edges(input_mask, size=10):
        input_mask[:size, :] = True
        input_mask[:, :size] = True
        input_mask[-size:, :] = True
        input_mask[:, -size:] = True
        return input_mask
    
    images = [inference_dir / model / image_id / 'pred_probability.tif' for model in modelnames]
    for image in images:
        if not image.exists():
            print(f'{image.as_posix()} does not exist')
            return None
    
    try:
        mean_image, out_meta_binary = calculate_mean_image(images)
    except:
        print(f'Read error of files {images}')
        return None
    
    for threshold in binary_threshold:

        # get binary object mask
        objects = (mean_image[0] >= threshold)
        
        # get individual numbered objects
        labels = label(objects, connectivity=2)
        
        # retrieve noData mask from image
        mask = np.array(np.isnan(mean_image[0]), np.uint8)
        #mask = np.array(np.isnan(mean_image), np.uint8)[0]
        # grow nodata mask around no data
        if CUCIM_AVAILABLE and try_gpu:
            cp.cuda.Device(gpu).use()
            dilated_mask = dilate_data_mask_gpu(mask, size=border_size)
        else:
            dilated_mask = dilate_data_mask(mask, size=border_size)
        # remove fixed sizes along image edges
        final_mask = mask_edges(dilated_mask, size=border_size)
        
        # get label ids which are in the edge region
        edge_labels = np.unique(labels * final_mask)
        # check which labels intersect with new mask
        isin_data_area = ~np.isin(labels, edge_labels)
        # remove labels intersecting new noData and create final binary mask
        out_binary = np.expand_dims((labels*isin_data_area > 0), 0)
        # remove small objects 32 px ~ 100m²
        out_binary = remove_small_objects(out_binary, min_size=minimum_mapping_unit)
        
        ### OUTPUTS
        # set paths
        thresh_str = str(threshold).replace('.','')
        
        
        outpath = inference_dir / ensemblename / image_id / f'{image_id}_{ensemblename}_class_{thresh_str}.tif'
        #outpath_class = Path(str(outpath).replace('proba', f'class_{thresh_str}'))
        outpath_shp = outpath.with_suffix('.gpkg')
        
        # write binary file
        os.makedirs(outpath.parent, exist_ok=True)
        with rasterio.open(outpath, 'w', **out_meta_binary, compress='deflate') as target:
            target.write(out_binary)
            
        # make vector
        s_polygonize = f'gdal_polygonize.py {outpath} -q -mask {outpath} -f "GPKG" {outpath_shp}'
        os.system(s_polygonize)
        
        if delete_binary:
            os.remove(outpath)
    
    
def load_and_parse_vector(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a given file path, reproject it to EPSG:4326,
    and parse image metadata from the file path to add as attributes.

    This function reads a GeoDataFrame from the specified file path, converts
    the GeoDataFrame's coordinate reference system to EPSG:4326, and parses
    the image ID from the parent directory name of the file path. It then
    extracts components from the image ID (take ID, tile ID, date, and satellite)
    and adds them as new columns in the GeoDataFrame.

    Args:
        file_path (str or pathlib.Path): Path to the vector file.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with added attributes representing
        parsed image metadata.

    Example:
        >>> file_path = '/path/to/your/vector_file.geojson'
        >>> parsed_gdf = load_and_parse_vector(file_path)
    """
    try:
        gdf = gpd.read_file(file_path).to_crs('EPSG:4326')
    except:
        print(f'Error on File: {file_path}')
        return None

    image_id = file_path.parent.name
    # parse and put into right format
    PSProductType = get_PS_products_type(image_id)
    if PSProductType == 'PSOrthoTile':
        take_id, tile_id, date, satellite = image_id.split('_')
    elif PSProductType == 'PSScene':
        if len(image_id.split('_')) == 4:
            date, take_id, _, satellite = image_id.split('_')
        else:
            date, take_id, satellite = image_id.split('_')
        
        date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        tile_id = None
        
    gdf['image_id'] = image_id
    gdf['take_id'] = take_id
    gdf['tile_id'] = tile_id
    gdf['date'] = date
    gdf['year'] = pd.to_datetime(gdf['date']).dt.year
    gdf['satellite'] = satellite
    
    return gdf



def create_ensemble_with_negative(inference_dir: Path, 
                                  modelnames: List[str],
                                  ensemblename: str, 
                                  image_id: str, 
                                  binary_threshold: list=[0.3,0.4,0.5], 
                                  negative_modelname: Optional[str] = None, 
                                  negative_binary_threshold: float=0.8,
                                  delete_proba: bool = True, 
                                  delete_binary: bool = True,
                                  erode_pixels: Optional[int] = None):
    """
    Calculate the ensemble prediction, including optional negative class, and write outputs to disk.
    
    Args:
        inference_dir (Path): Directory path for inference results.
        modelnames (List[str]): List of two model names for ensemble.
        ensemblename (str): Name of the ensemble model.
        image_id (str): ID of the image.
        binary_threshold (List[float], optional): List of binary threshold values.
        negative_modelname (str, optional): Name of the negative class model directory.
        negative_binary_threshold (float, optional): Binary threshold for negative class.
        delete_proba (bool, optional): Whether to delete the probability file after processing.
        delete_binary (bool, optional): Whether to delete binary files after creating vectors.
        erode_pixels (int, optional): Number of pixels for edge erosion.
    
    Returns:
        out_binary (numpy.ndarray): Binary result array.
    """
    # setup
    outpath = inference_dir / ensemblename / image_id / f'{image_id}_{ensemblename}_proba.tif'
    os.makedirs(outpath.parent, exist_ok=True)

    # calculate
    image1 = inference_dir / modelnames[0] / image_id / 'pred_probability.tif'
    image2 = inference_dir / modelnames[1] / image_id / 'pred_probability.tif'

    try:
        with rasterio.open(image1) as src1:
            with rasterio.open(image2) as src2:
                a1 = src1.read()
                a2 = src2.read()

            out_meta = src1.meta.copy()
            out_meta_binary = out_meta.copy()
            out_meta_binary['dtype'] = 'uint8'

    
        # calculate mean of all datasets
        out = np.mean([a1, a2], axis=0)

        if erode_pixels:
            #get mask
            mask = np.array(np.isnan(a1), np.uint8)[0]
            selem = disk(erode_pixels)
            # Apply erosion to the mask
            eroded_mask = dilation(mask, selem)

            #apply eroded_mask
            r, c = np.where(eroded_mask)
            out[0,r,c] = np.nan

            out[0, :erode_pixels, :] = np.nan
            out[0, -erode_pixels:, :] = np.nan
            out[0, :, :erode_pixels] = np.nan
            out[0, :, -erode_pixels:] = np.nan

        # optional if negative thresh
        if negative_modelname:
            image_negative = inference_dir / negative_modelname / image_id / 'pred_probability.tif'
            with rasterio.open(image_negative) as src_negative:
                a_negative = src_negative.read()
                a_binary = a_negative < negative_binary_threshold
                out *= a_binary

        with rasterio.open(outpath, 'w', **out_meta) as target:
            target.write(out)

        # write binary raster
        for threshold in binary_threshold:
            thresh_str = str(threshold).replace('.','')
            outpath_class = Path(str(outpath).replace('proba', f'class_{thresh_str}'))
            outpath_shp = outpath_class.with_suffix('.gpkg')

            out_binary = (out >= threshold)
            with rasterio.open(outpath_class, 'w', **out_meta_binary, compress='deflate') as target:
                target.write(out_binary)

            # make vector
            s_polygonize = f'gdal_polygonize.py {outpath_class} -q -mask {outpath_class} -f "GPKG" {outpath_shp}'
            os.system(s_polygonize)
            if delete_binary:
                os.remove(outpath_class)

        # delete proba file
        if delete_proba:
            os.remove(outpath)

        return out_binary

    except:
        return None