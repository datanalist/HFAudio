"""

Этот модуль позволяет создать произвольный аудио-датасет, отвечающий
требованиям datasets.load_dataset и csv-file к нему

Пример:
    $ python example_google.py

ЗАДАЧИ:
    * Дописать модуль для запуск в CLI и импортирования

"""

import os
from typing import List, Dict
from datasets import load_dataset
import pandas as pd


AUDIO_FOLDER = 'data/INPUT_ENG'


def transform_folder_name(folder_name: str) -> str:
    """Преобразует имя папки в валидную форму

    Args:
        folder_name (str): [имя папки]
    """
    folder_name = folder_name if folder_name[-1] == "/" else folder_name + "/"
    return folder_name


def get_files_from_folder(folder_path: str, is_relative: bool=False)-> List[str]:
    """Получить список файлов в каталоге
    Args:
        folder_path ([str]): [путь к каталогу]
        is_relative ([bool], optional): [выдать относительные пути к файлам]. Defaults to False:bool.

    Returns:
        List[str]: [список с файлами]
    """
    files_list = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if is_relative:
                files_list.append(file)
            else:
                files_list.append(os.path.join(root, file))
    return files_list



def create_dict_metadata(features:List[str], values:List[List[any]]) -> Dict[str, List[str]]:
    """Создать словарь с метаданными

    Args:
        columns ([List]): [columns for created metadata file]
        col_values ([List[List]]): [values of their columns]

    Returns:
        Dict[str, List[str]]: [словарь с метаданными]
    """
    len_for_padd = max([len(value) for value in values if value is not None])
    for val in range(len(values)):
        values[val] = values[val] if values[val] is not None else ["Empty"] * len_for_padd
    return {features[i]: values[i] for i in range(len(features))}


def create_metadatacsv(folder_path, metadata: Dict[List[str], List[any]]) -> str:
    """Создает файл с метаданными
    Args:
        folder_path ([str]): [путь, где будет лежать файл]
        Dict[List[str], List[any]]: [признак: значения]
        
    Returns:
        str: [путь до файла с метаданными]
    """
    df = pd.DataFrame(metadata)
    metadata_path = folder_path + "metadata.csv"
    if os.path.isfile(metadata_path):
        os.remove(metadata_path)
    df.to_csv(metadata_path, index=False, sep=",")
    assert os.path.isfile(metadata_path), "FIle was not created!"
    return metadata_path


def main(path_to_audio: str, 
                   feature_values: List[List[any]], 
                   is_realive: bool,
                   features: List[str]=["file_name"]):
    path_to_audio = transform_folder_name(path_to_audio)
    audio_files = get_files_from_folder(path_to_audio, is_realive)
    feature_values = [audio_files]
    metadata = create_dict_metadata(features, feature_values)
    # Создаем csv-file для датасета
    create_metadatacsv(AUDIO_FOLDER, metadata)





dataset = load_dataset("audiofolder", data_dir=AUDIO_FOLDER)


if __name__ == "__main__":
    pass
