import os
import argparse
def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("./data", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese", help="bert 预训练模型")
    parser.add_argument("--select_model_last", type=bool, default=0, help="选择模型 BertTextModel_last_layer")
    parser.add_argument("--class_num", type=int, default=15, help="分类数")
    parser.add_argument("--max_len", type=int, default=48, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learn_rate", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.5, help="失活率")
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    parser.add_argument("--num_filters", type=int, default=2, help="TextCnn 的卷积输出")
    parser.add_argument("--encode_layer", type=int, default=12, help="chinese bert 层数")
    parser.add_argument("--hidden_size", type=int, default=768, help="bert 层输出维度")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args