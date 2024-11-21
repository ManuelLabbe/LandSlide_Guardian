import os
import numpy as np
import configparser
def check_output_directory():
    output_dir = 'output'
    if not os.path.exists(output_dir):
        print("Creando directorio de salida 'output/'\n")
        os.makedirs(output_dir)
        
def check_soils_directory():
    soils_dir = 'Soils'
    if not os.path.exists(soils_dir):
        raise FileNotFoundError("El directorio 'Soils/' no existe\n")

def check_models_directory():
    soils_dir = 'models'
    if not os.path.exists(soils_dir):
        raise FileNotFoundError("El directorio 'models/' no existe\n")
    
def check_output_subdirectories():
    output_dirs = ['output/tif', 'output/csv']
    
    for dir_path in output_dirs:
        if not os.path.exists(dir_path):
            print(f"Creando directorio '{dir_path}/'")
            os.makedirs(dir_path)
            
def create_structure():
    print('===================================================================')
    print('Reading config')
    structu={}
    conf=configparser.ConfigParser()
    conf.read('config_RILEWS.conf')
    structu['conf']=conf
    print('\tDone')
    print('\n')
    return structu