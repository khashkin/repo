import logging
import os
import json
import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '.')


# извлечение последнего созданного .pkl файла
def loading_model() -> 'sklearn.pipeline.Pipeline':
    folder_path = f'{path}/data/models'
    files_list = os.listdir(folder_path)
    files_list = [os.path.join(folder_path, f) for f in files_list]
    files_list.sort(key=os.path.getmtime)
    models_path = f'{files_list[-1]}'
    number = models_path.split("_")[3]

    with open(models_path, 'rb') as file:
        model = dill.load(file)
    return model, number[:-4]


def save(result: pd.DataFrame, number: str) -> None:
    logging.info(f"{result[['id', 'pred']].to_string(index=False)}")
    result.to_csv(f'{path}/data/predictions/preds_{number}.csv', index=False)


def predict() -> None:
    frames = []
    files_name = os.listdir(f'{path}/data/test')

    model, number = loading_model()


    for name in files_name:
        with open(f'{path}/data/test/{name}') as file:
            file_json = json.load(file)
            df = pd.DataFrame.from_dict([file_json])

        y = model.predict(df)

        pred = {
            'id': file_json['id'],
            'pred': y[0]
        }

        frames.append(pred)

    result = pd.DataFrame(frames)
    save(result, number)


if __name__ == '__main__':
   predict()
