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
from datasets import Dataset, Audio
import pandas as pd


def transform_folder_name(folder_name: str) -> str:
    """Преобразует имя папки в валидную форму

    Args:
        folder_name (str): [имя папки]
    """
    folder_name = folder_name if folder_name[-1] == "/" else folder_name + "/"
    return folder_name


def get_files_from_folder(folder_path: str, is_metadata_csv_used: bool=False)-> List[str]:
    """Получить список файлов в каталоге
    Args:
        folder_path ([str]): [путь к каталогу]
        is_relative ([bool], optional): [выдать относительные пути к файлам]. 
            Defaults to False:bool.

    Returns:
        List[str]: [список с файлами]
    """
    files_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(("wav", "mp3")):
                if is_metadata_csv_used:
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
    for i, val in enumerate(values):
        values[i] = val if val is not None else ["Empty"] * len_for_padd
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


def create_audiodataset(path_to_audio: str,
                        feature_values: List[List[any]],
                        is_metadata_csv_used: bool,
                        feature_names: List[str]=["file_name"]) -> Dataset:
    """ создать произвольный аудио-датасет, отвечающий
        требованиям datasets.load_dataset и csv-file к нему

        Args:
            path_to_audio (str): [путь к аудио-файлу]
            feature_values (List[List[any]]): [значения именованных признаков]
            is_metadata_csv_used (bool): 
                [способ получения датасета: по csv-файлу или по meta-словарю]
            feature_names (List[str], optional): 
                [список признаков у аудио-файла]. Defaults to ["file_name"].

        Returns:
            [type]: [description]
    """
    sliced_path = transform_folder_name(path_to_audio)
    audio_files = get_files_from_folder(sliced_path, is_metadata_csv_used)
    if feature_values is None:
        feature_values = [audio_files]
    else:
        feature_values.extendleft(audio_files)
    metadata = create_dict_metadata(feature_names, feature_values)
    # Создаем csv-file для датасета
    create_metadatacsv(sliced_path, metadata)
    if is_metadata_csv_used:
        dataset = load_dataset("audiofolder", data_dir=path_to_audio)
    else:
        dataset = Dataset.from_dict(metadata).cast_column("file_name", Audio())
    return dataset


if __name__ == "__main__":
    AUDIO_FOLDER = 'data/INPUT_ENG'
    feature_names = ["file_name"]
    feature_values = None
    is_metadata_csv_used = False
    create_audiodataset(AUDIO_FOLDER,
                        feature_values,
                        is_metadata_csv_used,
                        feature_names)

