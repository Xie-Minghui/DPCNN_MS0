
from sklearn.model_selection import train_test_split
from src.config import config
from mindspore import Tensor
import mindspore


class ModelDataProcessor:
    def __init__(self):
        self.get_dict()

    def get_dict(self):
        self.word_dict = {}
        with open(config.vocab_file, 'r') as f:
            cnt = 0
            for line in f:
                line = line.rstrip()
                self.word_dict[line] = cnt
                cnt += 1

    def process_file(self, file_name:str):
        setences_list = []

        with open(file_name, 'r', encoding='Windows-1252') as f:
            for line in f:
                text = line.rstrip().split()
                setences_list.append(text)

        return setences_list


    def process_data(self, file_name_pos, file_name_neg):
        setences_list_pos = self.process_file(file_name_pos)
        setences_list_neg = self.process_file(file_name_neg)

        # 添加标签
        setences_list = setences_list_pos + setences_list_neg
        
        labels = [1 for i in range(len(setences_list_pos))] + [0 for i in range(len(setences_list_neg))]
        
        # 制作数据集
        X_train, X_test, y_train, y_test = train_test_split(setences_list, labels, test_size=0.3, shuffle=True, random_state=0, stratify=labels)

        return X_train, X_test, y_train, y_test

    def get_data(self):
        # 提供给训练文件获取分割好的数据集
        file_name_pos = './data/rt-polaritydata/pos.txt'
        file_name_neg = './data/rt-polaritydata/neg.txt'

        X_train, X_test, y_train, y_test = self.process_data(file_name_pos, file_name_neg)

        return X_train, X_test, y_train, y_test

    def get_data_loader(self):
        X_train, X_test, y_train, y_test = self.get_data()
        # 中间应该还增加对文本的编码
        train_text_ids = [[self.word_dict[word] for word in item] for item in X_train]
        test_text_ids = [[self.word_dict[word] for word in item] for item in X_test]
    
        return train_text_ids, test_text_ids, y_train, y_test

    def get_batch(self, x, y):
        assert len(x) == len(y) , "error shape!"

        n_batches = int(len(x) / config.batch_size)  # 统计共几个完整的batch
        for i in range(n_batches - 1):
            x_batch = x[i*config.batch_size: (i + 1)*config.batch_size]
            y_batch = y[i*config.batch_size: (i + 1)*config.batch_size]
            lengths = [len(seq) for seq in x_batch]
            max_length = max(lengths)
            for i in range(len(x_batch)):
                x_batch[i] = x_batch[i] + [0 for j in range(max_length-len(x_batch[i]))]

            yield x_batch, y_batch


if __name__ == '__main__':
    data_processor = ModelDataProcessor()
    X_train, X_test, y_train, y_test = data_processor.get_data_loader()
    for x_batch, y_batch in data_processor.get_batch(X_train, y_train):
        x_batch = Tensor(x_batch, mindspore.int32)
        y_batch = Tensor(y_batch, mindspore.int32)
        print(x_batch)
        print(y_batch)