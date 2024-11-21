import pickle
import os
import catboost
import numpy as np
import rasterio
from netCDF4 import Dataset
from osgeo import gdal, gdalconst, osr
from tqdm import tqdm
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import partial
import pandas as pd
from shapely.geometry import box
from rasterio.windows import from_bounds, bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
#from catboost import CatBoostClassifier

def features4model(file_path: str) -> list:
    #Features guardas en txt como [x1, x2, x3, ..., xn]
    with open(file_path, 'r') as file:
        return eval(file.read())

def load_model(model_path: str):
    # Cargar modelo
    # Es necesario importar la libreria a la que corresponde el modelo
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)
    
def load_models_and_features(model_path: str):
    # Carga el modelo y las caracteristicas a utilizar
    # output:
    #   model: sklearn.model_selection
    #   features: listv
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    features = model.best_estimator_.feature_names_
    return model, features

def load_data_tif(features: list):
    pass


def data_from_features_model(features: list, PATH = 'Soils/'):
    # Primero voy a crear una lista de todos los archivos que están dentro de cada carpeta de la
    # carpeta de Soils/
    # ej: Abrir Soils Abrir 'ROSETTA_MEAN' crear lista de todos los archivos dentro de 'ROSETTA_MEAN'
    
    # Se obtienen las carpetas dentro de SOILS/
    files = [f for f in os.listdir(PATH)]    
    
    # Ahora creo un diccionario donde a partir del nombre de la carpeta se obtengan todos los
    # tif que estan dentro de esta
    files_dict = {file: os.listdir(PATH + '/' + file) for file in files}
    # ahora de esta forma podemos pedir las features del modelo y verificamos a traves del diccionario
    # donde estan ubicada cada feature para de esta forma obtener los datos necesarios
    #path_file = []
    #features = ['PIRange_Bulkd.30-60cm.tif', 'alpha_5-15cm.tif']
    path_file = [k +'/'+i for k,v in files_dict.items() for i in v if i in features]
    print('-'*50)
    print(f'path de cada archivo: {path_file}\n')
    
    # Ahora a partir de la entrada features: list se obtendran los path necesarios para luego extraer
    # cada dato necesario de los tif
    
    # crear lista de paths, a cada feature le corresponde un path dentro de la lista paths
    # ciclo para abrir tif para cada paths
    
    
    return path_file

def tif_data_from_files_features(path_files: list[str], path_output_tif = 'output/output.tif', PATH= 'Soils/'):
    data_dict = {}
    with rasterio.open(path_output_tif) as src:
        meta = src.meta
        data = src.read(1)
        width = meta['width']
        height = meta['height']
        window = src.window(*src.bounds)
        data_window = src.read(1,window=window)
        print('Valores de window: ', len(data_window))
        data_dict = {'window_data': data_window}
        #print(f'Valores en window: {np.unique(data_window)}, largo de window: {len(data_window)}')
        for path in path_files:
            with rasterio.open(PATH + path) as src1:
                data = src1.read(1, window=window)
                print(f'Valores de data en {path}:  {np.unique(data)}, largo de {path}: {len(data)}')
                print(f'Cantidad de valores dentro de window: {data.size}')
                
                # Obtener frecuencia de cada valor
                # funcion sacada de https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
                # Descomentar para obtener la frecuencia de cada valor
                
                #unique_values, counts = np.unique(data, return_counts=True)
                #frequency_dict = dict(zip(unique_values, counts))
    
                #print(f'Frecuencia de cada valor: {frequency_dict}')
                #data_dict.update({PATH + path: data})
                
def tif_data_from_files_features(path_files: list[str], path_output_tif = 'Slope_SRTM_Zone_WGS84.tif', PATH= 'Soils/'):
    with rasterio.open(path_output_tif) as src:
        meta = src.meta
        base_data = src.read(1)
        window = src.window(*src.bounds)
        data_window = src.read(1,window=window)
        height = meta['height']
        width = meta['width']
        print(f'Height: {height}, Width: {width}')
        rows, cols = np.meshgrid(range(height), range(width), indexing='ij')
        xs, ys = rasterio.transform.xy(src.transform, rows.flatten(), cols.flatten())
        df = pd.DataFrame({
            'longitude': xs,
            'latitude': ys,
            'base_raster': data_window.flatten()
        })
        
        for path in tqdm(path_files, desc='Procesando archivos'):
            with rasterio.open(PATH + path) as file_src:
                data = file_src.read(1, window=window)
                df[path] = data.flatten()
        df.to_csv('output/data.csv', index=False)
    return df


def raster2vector(path_files: list[str], features: list[str], PATH = 'Soils/', PATH_SLOPE = 'Slope_SRTM_Zone_WGS84.tif', PATH_PP = 'datda.2024.01.01.tif'):
    # Pasar raster a vector
    
    with rasterio.open(PATH_SLOPE) as src_slope, rasterio.open(PATH_PP) as src_pp:
        ext1 = box(*src_slope.bounds)
        ext2 = box(*src_pp.bounds)
        intersection = ext1.intersection(ext2)
        window = from_bounds(*intersection.bounds, transform=src_pp.transform)
        data = src_slope.read(window=window)
        left, bottom, right, top = bounds(window, transform=src_pp.transform)
        new_transform, width, height = calculate_default_transform(src_slope.crs, src_pp.crs, data.shape[1], data.shape[0], left, bottom, right, top)
        height = window.height
        width = window.width
        kwargs = src_pp.meta.copy()
        kwargs.update({
            'width': width,
            'height': height
        })
        
def nc_to_tif(nc_file_path = 'WRFProno_20240908181246.nc'):
    # convertir archivo nc a tif
    nc_file = Dataset(nc_file_path, 'r')

    lon = nc_file.variables['XLONG'][:]
    lat = nc_file.variables['XLAT'][:]
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    #print(f'Latitud {lat_min} {lat_max}')
    #print(f'Longitud {lon_min} {lon_max}')

    cols, rows = len(lon[0]), len(lat[0])
    rainnc = nc_file.variables['RAINNC']
    cols, rows = np.size(rainnc[0], 1), np.size(rainnc[0], 0)
    dx = (lon_max - lon_min) / cols
    dy = (lat_max - lat_min) / rows
    transform = [lon_min, dx, 0, lat_min, 0, dy]

    
    string_date = []

    for time_index, timestamp_bytes in enumerate(nc_file.variables['Times']):
        timestamp_str = b''.join(timestamp_bytes).decode('utf-8')
        string_date.append(timestamp_str[:10])

    fec_uni = set(string_date)
    print('-'*50)
    #print(f'Fechas: {fec_uni}')
    for ind in fec_uni:
        ind_fec = [indice for indice, valor in enumerate(string_date) if valor == ind]
        #print(max(ind_fec), min(ind_fec))
        if min(ind_fec) > 0:
            data1 = rainnc[max(ind_fec), :, :] - rainnc[min(ind_fec)-1, :, :]
        else:
            data1 = rainnc[max(ind_fec), :, :]
        #print(data1.shape)
        
        output_tif_path = fr'output/{ind[:4]}_{ind[5:7]}_{ind[8:10]}.tif'    
        driver = gdal.GetDriverByName('GTiff')
        #print(f'rows, cols: {output_tif_path} {rows} {cols}')
        output_tiff = driver.Create(output_tif_path, cols,rows,  1, gdal.GDT_Float32)

        output_tiff.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        output_tiff.SetProjection(srs.ExportToWkt())
        #print('-'*20,np.flipud(data1))
        #data = np.flipud(data1) # No entiendo
        data = data1
        output_tiff.GetRasterBand(1).WriteArray(data)
        output_tiff = None
    nc_file.close()

def reproyectar_raster_PP(path_slope = 'Slope_SRTM_Zone_WGS84.tif', path_pp1 = 'output/2024_09_10.tif', path_name_output = 'output/output_pp.tif'):

    with rasterio.open(path_slope) as src1, rasterio.open(path_pp1) as src2:
        ext1 = box(*src1.bounds)
        ext2 = box(*src2.bounds)
        #print(f'ext1: {ext1}\next2: {ext2}')
        
        # intersection devuelve la geometría que se comparte entre las geometrías de entrada
        intersection = ext1.intersection(ext2)
        #print(f'intersection: {intersection}')
        
        window = from_bounds(*intersection.bounds, src1.transform)
        #print(f'Ventanas: {window}')
        
        # Leer los datos de la ventana de intersección del archivo de precipitación
        data_pp = src2.read(1, window=window)
        #print(f'Tipo de dato {type(data_pp)}\nDatos: {data_pp}')
        #print('bounds:', bounds(window, src1.transform))
        
        left, bottom, right, top = bounds(window, src1.transform)
        new_transform = src1.window_transform(window)
        height = window.height
        width = window.width
        #print(f'La ventana es: {width} X {height}, Teniendo: {window}')
        
        kwargs = src1.meta.copy()
        #print(f'Los kawrgs: {kwargs}')
        kwargs.update({
            'width': width,
            'height': height,
            'transform': new_transform,
            'crs': src2.crs  # Asegúrate de que el CRS sea el mismo que el de src2
        })
        #print(f'kwargs: {kwargs}')
        #print(window)
        
        # Creación del nuevo archivo TIFF con los datos de precipitación
        with rasterio.open(path_name_output, 'w', **kwargs) as dst:
            dst.write(data_pp, 1)
            
def reproyectar_raster_slope(path_slope = 'Slope_SRTM_Zone_WGS84.tif', path_pp = 'output/2024_09_12.tif', path_name_output = 'output/output.tif'):
    with rasterio.open(path_slope) as src1, rasterio.open(path_pp) as src2:
        ext1 = box(*src1.bounds)
        ext2 = box(*src2.bounds)
        #print(f'ext1: {ext1}\next2: {ext2}')
        # intersection devuelve la geometria que se comparte entre las geometrias de entradas
        intersection = ext1.intersection(ext2)
        #print(f'intersection: {intersection}')
        
        window = from_bounds(*intersection.bounds, src1.transform)
        #print(f'Ventanas: {window}')
        data = src1.read(window=window)
        #print(f'Tipo de dato {type(data)}\nDatos: {data}')
        #print('bounds:', bounds(window, src1.transform))
        left, bottom, right, top = bounds(window, src1.transform)
        #calculate default transform: (source coor, target coor, width, height, left, bot,right, top)
        #print('src1.crs',src1.crs)
        #print('src2.crs',src2.crs)
        # ---------
        #print('data.shape[1]', data.shape[1])
        #print('data.shape[2]', data.shape[2])
        new_transform, width, height = calculate_default_transform(src1.crs, src1.crs, data.shape[2], data.shape[1],
                                                                left, bottom, right, top)
        height = window.height
        width = window.width
        #new_transform, width, height = calculate_default_transform(src1.crs, src1.crs, width, height,
        #                                                           left, bottom, right, top)
        #print(f'La ventana es: {width} X {height}, Teniendo: {window}')
        kwargs = src1.meta.copy()
        #print(kwargs)
        kwargs.update({
            'width': width,
            'height': height
        })
        #print(f'kwargs: {kwargs}')
        #print(window)
        # Creación del nuevo path
        with rasterio.open(path_name_output, 'w+', **kwargs) as dst:
            print('------------------------------------------\n')
            reproject(
                source=data,
                destination=rasterio.band(dst, 1),
                src_transform=src1.window_transform(window),
                window=window,
                src_crs=src1.crs,
                dst_crs=src2.crs,
                dst_transform=new_transform,
                resampling=Resampling.bilinear
            )

def sample_random_indices(rows: np.ndarray, cols: np.ndarray, n_samples: int = 10) -> tuple:
    total_points = len(rows.flatten())
    random_indices = np.random.choice(total_points, n_samples, replace=False)
    sampled_rows = rows.flatten()[random_indices]
    sampled_cols = cols.flatten()[random_indices]
    
    return sampled_rows, sampled_cols
    
def tif_to_dataframe_with_window_parallel(reference_tif: str, tif_paths: list[str]):
    def process_tif(args):
        tif_path, longs, lats, ref_crs = args
        with rasterio.open(tif_path) as src:
            if src.crs != ref_crs:
                raise ValueError(f"El CRS de {tif_path} no coincide con el del TIF de referencia.")
            values = np.array([x[0] if x else np.nan for x in src.sample(zip(longs, lats))])
            return tif_path.split('/')[-1], values

    with rasterio.open(reference_tif) as ref_src:
        
        meta = ref_src.meta
        ref_array = ref_src.read(1)
        ref_transform = ref_src.transform
        height = meta['height']
        width = meta['width']
        print('Realizando meshgrid...')
    
        # Siguiente linea: función para sólo usar una fracción de los datos para testear la función:
        # adapta aleatoriamente n_samples a partir de rows, cols, modoficando rows, cols
        # comentantar lienea sample_random_indices para usar todos los puntos
        rows, cols = np.meshgrid(range(height), range(width), indexing='ij')
        rows, cols = sample_random_indices(rows, cols, n_samples=100000)
        
        print(f'rows: {rows.flatten()}\ncols: {cols.flatten()}')
        print('-'*50)
        print(f'rows: {len(rows.flatten())}\ncols: {len(cols.flatten())}')
        print('\nRealizando transform desde xy...')
        xs, ys = rasterio.transform.xy(ref_src.transform, rows.flatten(), cols.flatten())
        longitudes = np.array(xs)
        latitudes = np.array(ys)
        
        # Se agreega el path de slope y PP dentro de los datos
        all_paths = ['output/output.tif', 'output/output_pp.tif'] + tif_paths
        
        args = [(path, longitudes, latitudes, ref_src.crs) for path in all_paths]
        # Para ocupar con todas las cpu disponibles menos 1
        with ThreadPoolExecutor(max_workers=mp.cpu_count()-1) as executor:
            results = list(tqdm(
                executor.map(lambda x: process_tif(x), args),
                total=len(all_paths),
                desc="Procesando archivos"
            ))
        
        data = {
            'Latitude': latitudes,
            'Longitude': longitudes,
            **dict(results)
        }
        
        return pd.DataFrame(data)
    
def df_in_model_to_proba(df: pd.DataFrame, model):
    X = df.drop(columns=['Latitude', 'Longitude'])
    proba = model.best_estimator_.predict_proba(X)
    return proba

def proba_in_df(df: pd.DataFrame, proba: np.array) -> pd.DataFrame:
    df['Probabilidad'] = proba[:, 1]
    return df

def save_df_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def proba_to_tif(df: pd.DataFrame, reference_tif: str, path_output: str):
    with rasterio.open(reference_tif) as ref_src:
        meta = ref_src.meta
        height = meta['height']
        width = meta['width']
        proba = df['Probabilidad'].values
        proba = proba.reshape(height, width)
        with rasterio.open(path_output, 'w', **meta) as dst:
            dst.write(proba, 1)

    
"""if __name__ == '__main__':
    model, features = load_models_and_features('models/catboost_model_random_search.pkl')
    print(f'Modelo: {type(model)}\n')
    print(f'Caracteristicas: {features}\n')
    print(f'Cantidad de caracteristcas: {len(features)}')
    
    # testear data_from_features_model
    path_files = data_from_features_model(features=features)
    print(type(path_files))
    print(f'Los path de todos los files {path_files}')
    
    tif_data_from_files_features(path_files)"""
    # paso 2
    # cargar datos de variables X de modelo entrenado, considerar raster de humedad como valor de 0.3 y cargar un rasteer de pp aleatoreo con nombre pp1.tif
    
    
    # paso 3
    # pasar raster a vectores *(X)
    
    
    # paso 4
    # fitear con modelo de ML *( model.predict  proba)
    
    # paso 5
    # pasar proba de Y a matriz como reshape
    
    # paso 6
    # guardar matriz como geotiff usar quizas rasterio
    
    # paso 7
    # tomar un cafe