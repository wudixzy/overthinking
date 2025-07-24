import json
import jsonlines
import io
from openai import OpenAI
from datasets import Dataset


def read_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as r_file:
        datas = json.load(r_file)
    r_file.close()
    return datas


def read_json_file_line(file_path):
    datas = []
    with io.open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            datas.append(json_object)
    return datas


def save_file(datas, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(datas, file, indent=4, ensure_ascii=False)
    file.close()


def write_file(data, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data))
        file.write('\n')
    file.close()


def read_jsonl_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        datas = list(jsonlines.Reader(file))
    file.close()
    return datas


def ask_gpt(prompts: str, model, url, api_key):
    client = OpenAI(
        base_url=url,
        api_key=api_key
    )

    response = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "user", "content": prompts},

      ]
    )
    response = response.choices[0].message.content
    return response


def dict_list_to_hf_dataset(dict_list):
    """
    将字典列表转换为HuggingFace Dataset
    :param dict_list: 字典列表，每个字典代表一个样本
    :return: datasets.Dataset对象
    """
    # 将字典列表转换为字典的字典
    feature_dict = {key: [] for key in dict_list[0].keys()}
    for item in dict_list:
        for key, value in item.items():
            feature_dict[key].append(value)

    # 创建Dataset
    dataset = Dataset.from_dict(feature_dict)
    return dataset
