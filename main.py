from model_raster import *
from utils import *

if __name__ == '__main__':
    # Se verifica que los directorios existan
    check_output_directory()
    check_output_subdirectories()
    check_soils_directory()
    #check_models_directory()
    
    # Se extraern las features que usa el modelo
    model, features = load_models_and_features('models/catboost_model_random_search.pkl')
    
    # Se obtiene el path de los archivos que se van a usar
    path_files = data_from_features_model(features=features)
    
    # Tranformación de nc a tif
    nc_to_tif(nc_file_path = 'WRFProno_20240908181246.nc')
    
    # Se reproyectan los rasters
    reproyectar_raster_slope(path_slope = 'Slope_SRTM_Zone_WGS84.tif', path_pp='output/2024_09_10.tif', path_name_output = 'output/output.tif')
    reproyectar_raster_PP(path_slope = 'Slope_SRTM_Zone_WGS84.tif', path_pp1 = 'output/2024_09_10.tif', path_name_output = 'output/output_pp.tif')
    
    # Se obtiene el dataframe con las features
    reference_tif = 'output/output.tif'
    df_output = 'model_data_features.csv'
    df = tif_to_dataframe_with_window_parallel(reference_tif, path_files)
    save_df_to_csv(df = df, path = df_output)
    
    # Obtener las probabilidades con el modelo
    proba = df_in_model_to_proba(df , model)
    df = proba_in_df(df, proba)
    save_df_to_csv(df = df, path = df_output)
    proba_to_tif(df = df, reference_tif = reference_tif, path_output = 'output/tif/probability.tif')
