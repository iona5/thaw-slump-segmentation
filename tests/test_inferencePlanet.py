import shutil, pytest
from pathlib import Path

from thaw_slump_segmentation.scripts import inference

expected_inference_files = {
    "ndvi.jpg",
    "planet.jpg",
    "pred_binarized.dbf",
    "pred_binarized.gpkg",
    "pred_binarized.jpg",
    "pred_binarized.prj",
    "pred_binarized.shp",
    "pred_binarized.shx",
    "pred_binarized.tif",
    "pred_probability.jpg",
    "pred_probability.tif",
    "relative_elevation.jpg",
    "slope.jpg",
    "tcvis.jpg",
}

@pytest.mark.parametrize(
        "planet_id",
         ["20230807_191418_31_241d", "5825117_1072218_2022-07-31_2481"],
         ids=["Orthotile","Scene"])
def testInferencePlanetTile(data_dir, tmp_path:Path, gdal_bin, gdal_path, planet_id, gpu_id):

    target_dir = tmp_path / "intermediate"
    input_dir = target_dir / "tiles"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    shutil.copytree(data_dir / "intermediate" / "tiles" / planet_id, input_dir / planet_id)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

    inference.inference(
        name = False, # just write folder to output_dir
        model_path = data_dir / "models/RTS_v6_tcvis",
        tile_to_predict=[planet_id], # has to be a list
        gdal_path=gdal_path, gdal_bin=gdal_bin,
        n_jobs=1, data_dir=target_dir,
        log_dir=log_dir, 
        inference_dir=output_dir
    )

    assert (output_dir / planet_id ).exists()

    # check if all of the expected files are there (and only those)
    files = set( [f.name for f in (output_dir / planet_id ).iterdir()]  )

    assert files == expected_inference_files



