import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
from pyjarowinkler import distance  # pip install pyjarowinkler


class ali_process:
    def __init__(self, data_url):
        self.meta = pd.read_csv(data_url, sep="\x01", encoding="utf-8", header=None)
        self.meta_drop_null = self.drop_null()
        self.ali = ["Brand Name", "Material", "Color", "Category"]
        self.attribute_count = self.meta_drop_null.iloc[:, 1].value_counts()

    def topk_name(self, k):
        topk = []
        count = 0
        for index, value in self.attribute_count.items():
            if count < k:
                topk.append(index)
            count += 1
        return topk

    def show_topk(self, k):
        count = 0
        for index, value in self.attribute_count.items():
            if count < k:
                print(index + ": " + str(value))
                count += 1
            else:
                break

    def drop_null(self):
        meta_copy = self.meta.copy()
        meta_copy.dropna(inplace=True)
        meta_copy.reset_index(drop=True, inplace=True)
        return meta_copy

    def get_topk(self, k, drop_null=True, bio=True):
        """
        return top k frequent attributes and the rows contain these attributes
        Args:
            k:

        Returns:

        """
        topk = self.topk_name(k)
        if drop_null:
            is_topk = [True if self.meta_drop_null.iloc[i, 1] in topk else False for i in range(self.meta_drop_null.iloc[:, 1].size)]
            meta_topk = self.meta_drop_null[is_topk]
        else:
            is_topk = [True if self.meta.iloc[i, 1] in topk else False for i in range(self.meta.iloc[:, 1].size)]
            meta_topk = self.meta[is_topk]
        if bio:
            meta_topk = self.to_bio(meta_topk, "top"+str(k))
        return meta_topk

    def get_ali(self, drop_null=True, bio=True):
        """
        ali use four attributes: "Brand Name", "Material", "Color", "Category".
        Args:
            drop_null: default is True to drop the null rows
            to_bio: convert to bio and save to text file

        Returns:
            return ali dataset
        """
        if drop_null:
            is_ali = [True if self.meta_drop_null.iloc[i, 1] in self.ali else False for i in range(self.meta_drop_null.iloc[:, 1].size)]
            meta_ali = self.meta_drop_null.iloc[is_ali]
        else:
            is_ali = [True if self.meta.iloc[i, 1] in self.ali else False for i in range(self.meta.iloc[:, 1].size)]
            meta_ali = self.meta.iloc[is_ali]
        if bio:
            meta_ali = self.to_bio(meta_ali, "Ali")
        return meta_ali

    def str_process(self, name):
        # rm leading blank
        name = name.lstrip()
        # rm tail blank
        name = name.rstrip()
        # rm \t
        name = name.strip("\t")
        # rm \n
        name = name.strip("\n")
        # remove duplicate spaces
        name = re.split(' |/|-|;|#|&|,', name)
        name = [re.sub('[^A-Za-z0-9]+', '', name_value) for name_value in name]
        name_clean = []
        for value in name:
            if value != '':
                name_clean.append(value.lower())
        return name_clean

    def similar(self, val, n_gram):
        """
        calculate the similarity of two string list
        Args:
            val:
            n_gram:

        Returns:

        """
        score = 0
        for val_value, n_gram_value in zip(val, n_gram):
            score += distance.get_jaro_distance(val_value, n_gram_value, winkler=True, scaling=0.1)
        return score/len(val)

    def n_gram_match(self, name, val):
        """
        find the matched part in name which is same with attr
        Args:
            name: name list
            attr: attr list

        Returns:
            a list of start and end index of the matched part [(s, n), (s, n), ...]
        """
        val_len = len(val)
        matched_index = []
        for index in range(len(name)):
            # print(index)
            if index <= len(name) - val_len:
                if self.similar(val, name[index: index + val_len]) > 0.9:
                    matched_index.append((index, index + val_len))
        return matched_index

    def to_bio(self, meta, dataset_name):
        """
        convert datafram data to bio format and save the statistical info
        Args:
            meta: the dataset to be converted
            name: the name of the meta dataset

        Returns:
            save bio to text file
        """
        groups = meta.groupby(0)
        # each group has the same sentence. The # of rows represents the number of attributes
        attr_num = [0 for i in range(groups.ngroups)]
        i = 0
        for name, group in groups:
            # store the number of rows in each group in attr_num
            attr_num[i] = (group.iloc[:, 0]).size
            # test
            # print("group " + str(i))
            name = self.str_process(name)
            #create a tuple for each token (token, [BIO flag])
            token_tuple = [(token, []) for token in name]
            # within_group_index = 0
            for _, attr_value in group.iterrows():
                # print("within_group_index: " + str(within_group_index))
                attr = str(attr_value.iloc[1])
                # remove duplicate spaces
                # val_clean = " ".join(str(attr_value.iloc[2]).split())
                val = str(attr_value.iloc[2]).split()
                # val = val_clean.split()
                matched_index = self.n_gram_match(name, val)
                if len(matched_index) > 0:
                    for (m_start, m_end) in matched_index:
                        for j in range(m_start, m_end):
                            if j == m_start:
                                token_tuple[j][1].append("B-" + attr)
                            else:
                                token_tuple[j][1].append("I-" + attr)
                # within_group_index += 1
            if i < groups.ngroups * 0.8:
                with open("../data/Ali/" + dataset_name + "-train.iob2", "a+") as f:
                    # after matching, add "O" if there is no matching
                    for tul in token_tuple:
                        if len(tul[1]) == 0:
                            tul[1].append("O")
                        f.write(tul[0] + "\t" + '\t'.join(tul[1]) + "\n")
                    f.write("\n")
            elif i < groups.ngroups * 0.9:
                with open("../data/Ali/" + dataset_name + "-dev.iob2", "a+") as f:
                    # after matching, add "O" if there is no matching
                    for tul in token_tuple:
                        if len(tul[1]) == 0:
                            tul[1].append("O")
                        f.write(tul[0] + "\t" + '\t'.join(tul[1]) + "\n")
                    f.write("\n")
            else:
                with open("../data/Ali/" + dataset_name + "-test.iob2", "a+") as f:
                    # after matching, add "O" if there is no matching
                    for tul in token_tuple:
                        if len(tul[1]) == 0:
                            tul[1].append("O")
                        f.write(tul[0] + "\t" + '\t'.join(tul[1]) + "\n")
                    f.write("\n")
            i += 1
        # sort ascending order
        attr_num.sort()
        # get the frequency
        attr_num_s = pd.Series(attr_num)
        attr_num_g = attr_num_s.value_counts()
        indexs, values =[], []
        for index, value in attr_num_g.items():
            indexs.append(index)
            values.append(value)
        plt.plot(indexs, values)
        plt.ylabel("Frequency")
        plt.xlabel("# of attrs in sentence")
        plt.title(dataset_name)
        plt.savefig('../data/Ali/' + dataset_name + '.png')
        plt.clf()
        return i


class CoNLL2003_process:
    def __init__(self, data_url):
        self.data_url = data_url
        self.meta = pd.read_csv(data_url, sep=" ", encoding="utf-8", header=None, skip_blank_lines=False, quoting=3)
        self.meta_bio = self.meta[[0, 3]]

    def save_ner(self, file_name):
        folder_path = os.path.dirname(self.data_url)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "a+") as f:
            for index, row in self.meta_bio.iterrows():
                if str(row[0]) == "nan":
                    line = "\n"
                else:
                    line = str(row[0]) +"\t" + str(row[3]) + "\n"
                f.write(line)


def main():
    # ali_data = ali_process('../data/Ali/publish_data.txt')
    # ali_data.show_topk(5)
    # row_num = ali_data.get_ali()
    # ali_data.get_topk(10)
    # print("There are " + str(row_num) + "rows")
    CoNLL2003_train = "../data/CoNLL2003/eng.train"
    CoNLL2003_train_ner = "conll2003_train.bio"
    CoNLL2003_dev = "../data/CoNLL2003/eng.testa"
    CoNLL2003_dev_ner = "conll2003_dev.bio"
    CoNLL2003_test = "../data/CoNLL2003/eng.testb"
    CoNLL2003_test_ner = "conll2003_test.bio"
    CoNLL2003_train_obj = CoNLL2003_process(CoNLL2003_train)
    CoNLL2003_train_obj.save_ner(CoNLL2003_train_ner)
    CoNLL2003_dev_obj = CoNLL2003_process(CoNLL2003_dev)
    CoNLL2003_dev_obj.save_ner(CoNLL2003_dev_ner)
    CoNLL2003_test_obj = CoNLL2003_process(CoNLL2003_test)
    CoNLL2003_test_obj.save_ner(CoNLL2003_test_ner)


if __name__ == "__main__":
    main()
