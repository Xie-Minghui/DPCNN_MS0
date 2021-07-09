# import os, sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # __file__获取执行文件相对路径，整行为取上一级的上一级目录
# sys.path.append(BASE_DIR)

class Config:

    def __init__(self,
            embedding_dim=300,
        ):
        
        self.embedding_dim = embedding_dim
        
        self.num_filter = 32
        self.num_rel = 2
        self.batch_size = 128
        self.vocab_file = './data/vocab.txt'

        cnt = 1  # 添加pad的位置
        with open(self.vocab_file, 'r') as f:
            for line in f:
                cnt += 1

        self.vocab_size = cnt
        self.epochs = 20
        self.lr = 1e-3

config = Config()

