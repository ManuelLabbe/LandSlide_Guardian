from model_raster import *
from utils import *
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """
    En donde están cargándose los puntos es en la función 
    fast_extract_raster_values(reference_tif, path_files)
    a partir de las variables Height, Width obtenidas de reference_tif()
    el cual corresponde a output.tif el que es el raste Slope después
    de pasar por la punción reproyectar_raster_slope()
    
    En /output/csv/ está el df con prob sólo con algunos puntos para
    testear que funcionara bien
    """
    # Obtención de todo el path de los archivos a partir del archivo de configuración
    structu = create_structure()
    resolution = float(structu['conf']['output']['resolution'])
    path_model = structu['conf']['ML']['path']
    path_nc = structu['conf']['NC']['lc']
    path_slope_data = structu['conf']['data_source']['slope_data']
    path_soil_data = structu['conf']['output']['path_soil_data']
    
    # Path de los outputs
    output_dir = structu['conf']['output']['folder_out']
    #path_df_output_w_proba = f'{output_dir}csv/df_w_prob.csv'
    #path_raster_proba = f'{output_dir}tif/probability/raster_proba.tif'
    #path_output_pp = f'{output_dir}tif/output_pp.tif'
    #path_output_slope = f'{output_dir}tif/output_slope.tif'
    
    # Se verifica que los directorios existan
    # En caso de los output se crean los subdirectorios
    # En caso de soils se envía mensaje de error de configruación
    check_output_directory(output_dir)
    check_output_subdirectories(output_dir)
    check_soils_directory(path_soil_data)
    #check_models_directory()
    
    # Se extraern las features que usa el modelo
    model, features = load_models_and_features(path_model)
    
    # Se obtiene el path de los archivos que se van a usar
    path_files = data_from_features_model(features=features)
    path_files = [path_soil_data + path for path in path_files]
    # Tranformación de nc a tif
    nc_to_tif(nc_file_path = path_nc)
    
    for file in tqdm(list_files_from_tif_nc_pp(output_dir), desc='Procesando archivos de cada fecha'):
        date_file = file.split('/')[-1].replace('.tif', '')
        path_df_output_w_proba = f'{output_dir}csv/df_w_prob_{date_file}.csv'
        path_raster_proba = f'{output_dir}tif/probability/raster_proba_{date_file}.tif'
        path_output_pp = f'{output_dir}tif/output_pp_{date_file}.tif'
        path_output_slope = f'{output_dir}tif/output_slope_{date_file}.tif'
        # Se reproyectan los rasters
        reproyectar_raster_slope(path_slope = path_slope_data, 
                                path_pp= file, 
                                path_name_output = path_output_slope)
        
        reproyectar_raster_PP(path_slope = path_slope_data, 
                            path_pp1 = file, 
                            path_name_output = path_output_pp)
        
        # Se obtiene el dataframe con las features
        reference_tif = path_output_pp
        #df_output = 'output/csv/model_data_features.csv'
        path_files_w_slope_pp = [path_output_pp, path_output_slope] + path_files
        # Se estan utilizando solo algunos puntos para probar las funciones y
        # mayor rapidez, revisar en model_raster.py funcion tif_to_data_frame_with_window_parallel()
        df = fast_extract_raster_values(reference_tif, path_files_w_slope_pp, resolution=resolution)
        output_slope = path_output_slope.split('/')[-1]
        output_pp = path_output_pp.split('/')[-1]
        df = df.rename(columns={
            output_slope: 'slope',
            output_pp: 'PP'
        })
        # =============================================================================
        # A continuación se crean estos para simulación y obtener resultados
        # Por ahora se está creando aleatoriamente el valor
        
        #Aquí se crea el valor_humedad_suelo de forma normal con media 0.2
        n_samples = len(df)
        random_values = np.random.normal(loc=0.2, scale=0.1, size=n_samples)
        df.insert(2, 'valor_humedad_suelo1', random_values)
        df['slope'] = df['slope'].replace(-9999, np.nan)
        df['PP'] = df['PP'].replace(-9999, np.nan)
        df = df.fillna(df.mean())
        # =============================================================================
        
        proba = df_in_model_to_proba(df , model)
        df = proba_in_df(df, proba)
        save_df_to_csv(df = df, path = path_df_output_w_proba)
        # df con probabilidad guardado en /output/csv/model_data_features.csv
        print('-'*50)
        print('df Guardado con probabilidad')
        print('-'*50)
        points_to_raster(df,output_path=path_raster_proba, resolution=resolution)
        #clear_directory(output_dir)
    