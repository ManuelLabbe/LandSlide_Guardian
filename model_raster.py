import pickle
import os
import catboost
import numpy as np
import rasterio
from tqdm import tqdm
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
    # PREGUNTAR: como obtengo los pixeles (o lat lon)
    
    
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
    # Open base raster to get metadata and coordinates
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
        

    
if __name__ == '__main__':
    model, features = load_models_and_features('models/catboost_model_random_search.pkl')
    print(f'Modelo: {type(model)}\n')
    print(f'Caracteristicas: {features}\n')
    print(f'Cantidad de caracteristcas: {len(features)}')
    
    # testear data_from_features_model
    path_files = data_from_features_model(features=features)
    print(type(path_files))
    print(f'Los path de todos los files {path_files}')
    
    tif_data_from_files_features(path_files)
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