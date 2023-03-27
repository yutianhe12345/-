import jieba
import math
import time
import os
import re
from collections import Counter


class ReadFile:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_corpus(self):
        text_list = []
        r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        listdir = os.listdir(self.root_dir)
        characters_count = 0
        for file_name in listdir:
            path = os.path.join(self.root_dir, file_name)
            if os.path.isfile(path) and file_name.split('.')[-1] == 'txt':
                with open(os.path.abspath(path), "r", encoding='ansi') as file:
                    print('file:', file)
                    file_content = file.read()
                    file_content = re.sub(r1, '', file_content)
                    file_content = file_content.replace("\n", '')
                    file_content = file_content.replace(" ", '')
                    file_content = file_content.replace('\u3000', '')
                    file_content = file_content.\
                        replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
                    characters_count += len(file_content)
                    text_list.append(file_content)
            elif os.path.isdir(path):
                print('文件路径不存在!!!!!!')
        return text_list, characters_count


def calculate_single_character_entropy(corpus):
    before = time.time()
    split_characters = []
    for text in corpus:
        split_characters += [char for char in text]
    characters_count = len(split_characters)
    single_character_dict = dict(Counter(split_characters))
    print("语料库字数:", characters_count)

    entropy = []
    for _, value in single_character_dict.items():
        entropy.append(-(value / characters_count) * math.log(value / characters_count, 2))
    print("基于字的中文信息熵为:", round(sum(entropy), 5))
    after = time.time()
    print("运行时间:", round(after - before, 5), "s")


def calculate_unary_entropy(corpus, characters_count):
    before = time.time()
    split_words = []
    for text in corpus:
        split_words += list(jieba.cut(text))
    words_count = len(split_words)
    unary_words_dict = dict(Counter(split_words))

    print("语料库字数:", characters_count)
    print("一元词组数量:", words_count)
    print("平均词长:", round(characters_count / words_count, 5))

    entropy = []
    for _, value in unary_words_dict.items():
        entropy.append(-(value / words_count) * math.log(value / words_count, 2))
    print("基于词的一元模型的中文信息熵为:", round(sum(entropy), 5))
    after = time.time()
    print("运行时间:", round(after - before, 5), "s")


def calculate_binary_entropy(corpus, characters_count):
    before = time.time()
    all_words = []
    binary_words_dict = {}
    for text in corpus:
        split_words = list(jieba.cut(text))
        for i in range(len(split_words) - 1):
            binary_words_dict[(split_words[i], split_words[i + 1])] = binary_words_dict.get(
                (split_words[i], split_words[i + 1]), 0) + 1
        all_words += split_words

    words_count = len(all_words)
    unary_words_dict = dict(Counter(all_words))

    print("语料库字数:", characters_count)
    print("一元词组数量:", words_count)
    print("平均词长:", round(count / words_count, 5))

    binary_words_count = sum([value for _, value in binary_words_dict.items()])
    print("二元词组数量:", binary_words_count)

    entropy = []
    for key, value in binary_words_dict.items():
        joint_probability_xy = value / binary_words_count  # 计算联合概率p(x,y)
        conditional_probability_x_y = joint_probability_xy / (unary_words_dict[key[0]]/words_count)  # 计算条件概率p(x|y)
        entropy.append(-joint_probability_xy * math.log(conditional_probability_x_y, 2))  # 计算二元模型的信息熵
    print("基于词的二元模型的中文信息熵为:", round(sum(entropy), 5))

    after = time.time()
    print("运行时间:", round(after - before, 5), "s")


def calculate_ternary_entropy(corpus, characters_count):
    before = time.time()
    all_words = []
    binary_words_dict = {}
    ternary_words_dict = {}
    for text in corpus:
        split_words = list(jieba.cut(text))
        for i in range(len(split_words) - 1):
            binary_words_dict[(split_words[i], split_words[i + 1])] = binary_words_dict.get(
                (split_words[i], split_words[i + 1]), 0) + 1
        for i in range(len(split_words) - 2):
            ternary_words_dict[((split_words[i], split_words[i + 1]), split_words[i + 2])] = ternary_words_dict.get(
                ((split_words[i], split_words[i + 1]), split_words[i + 2]), 0) + 1
        all_words += split_words
    words_count = len(all_words)
    unary_words_dict = dict(Counter(all_words))

    print("语料库字数:", characters_count)
    print("一元词组数量:", words_count)
    print("平均词长:", round(characters_count / words_count, 5))

    binary_words_count = sum([value for _, value in binary_words_dict.items()])
    print("二元词组数量:", binary_words_count)
    ternary_words_count = sum([value for _, value in ternary_words_dict.items()])
    print("三元词组数量:", ternary_words_count)

    entropy = []
    for key, value in ternary_words_dict.items():
        joint_probability_xyz = value / ternary_words_count  # 计算联合概率p(x,y,z)
        conditional_probability_x_yz = joint_probability_xyz / (binary_words_dict[key[0]] / binary_words_count)  # 计算条件概率p(x|y,z)
        entropy.append(-joint_probability_xyz * math.log(conditional_probability_x_yz, 2))  # 计算三元模型的信息熵
    print("基于词的三元模型的中文信息熵为:", round(sum(entropy), 5))

    after = time.time()
    print("运行时间:", round(after - before, 5), "s")


def delete_stop_words(stop_words_path, corpus):
    before = time.time()
    with open(stop_words_path, 'r', encoding='utf-8') as stop_words_file:
        stop_words = [line.strip() for line in stop_words_file.readlines()]

    new_corpus = []
    character_count = 0
    for text in corpus:
        new_words = []
        split_words = list(jieba.cut(text))
        for word in split_words:
            if word not in stop_words:
                new_words.append(word)
        character_count += len(''.join(map(str, new_words)))
        new_corpus.append(''.join(map(str, new_words)))

    after = time.time()
    print("删除停词运行时间:", round(after - before, 5), "s")
    return new_corpus, character_count


if __name__ == '__main__':
    read_file = ReadFile("./jyxstxtqj_downcc.com")
    text_list, count = read_file.get_corpus()

    """calculate_single_character_entropy(corpus=text_list)
    calculate_unary_entropy(corpus=text_list, characters_count=count)
    calculate_binary_entropy(corpus=text_list, characters_count=count)
    calculate_ternary_entropy(corpus=text_list, characters_count=count)"""

    new_text_list, new_count = delete_stop_words(stop_words_path='./停词表.txt', corpus=text_list)

    calculate_single_character_entropy(corpus=new_text_list)
    calculate_unary_entropy(corpus=new_text_list, characters_count=new_count)
    calculate_binary_entropy(corpus=new_text_list, characters_count=new_count)
    calculate_ternary_entropy(corpus=new_text_list, characters_count=new_count)




