import os
import shutil
import numpy as np
import configparser

def create_structure():
    print('===================================================================')
    print('Reading config')
    structu={}
    conf=configparser.ConfigParser()
    conf.read('config_RILEWS.conf')
    structu['conf']=conf
    print('\tDone')
    print('===================================================================')
    return structu

def check_output_directory(output_dir):
    output_dir = os.path.normpath(output_dir)
    print(f'Verificando directorio de salida {output_dir}')
    if not os.path.exists(output_dir):
        print("Creando directorio de salida 'output/'\n")
        os.makedirs(output_dir)
        
def check_soils_directory(soils_dir):
    soils_dir = os.path.normpath(soils_dir)
    print(f"Verificando directorio de suelos {soils_dir}")
    if not os.path.exists(soils_dir):
        raise FileNotFoundError(f"El directorio {soils_dir} no existe\nRevisar el archivo de configuración 'config_RILEWS.conf'\n")

def check_models_directory():
    soils_dir = 'models'
    if not os.path.exists(soils_dir):
        raise FileNotFoundError("El directorio 'models/' no existe\n")
    
def check_output_subdirectories(output_dir):
    print(f'Verificando subdirectorios de salida en {output_dir}')
    output_dir = os.path.normpath(output_dir)
    output_dirs = [f'{output_dir}/tif', f'{output_dir}/csv', f'{output_dir}/tif_nc_pp', f'{output_dir}/tif/probability']
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            print(f"Creando directorio '{dir_path}/'")
            os.makedirs(dir_path)
            
def list_files_from_tif_nc_pp(output_dir):
    directory = os.path.normpath(output_dir)
    directory = directory + '/tif_nc_pp'
    files = os.listdir(directory)
    files = [f'{directory}/{file}' for file in files]
    return files

def clear_directory(output_dir):
    directory = os.path.normpath(output_dir)
    directory = directory + '/tif_nc_pp'
    print(f'Limpiando directorio {directory}')
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Error al intentar borrar {file_path}. Razón: {e}')