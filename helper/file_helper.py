import requests
import json
import csv

def download_image(source, filename):
    with open(filename, 'wb') as f:
        image_content = requests.get(source).content
        f.write(image_content)
        print('Download image at: ', filename)


def write_txt(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(data)
    return True


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return True


def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_csv(filename, data):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        for row in data:
            csv_writer.writerow(row)
    return True