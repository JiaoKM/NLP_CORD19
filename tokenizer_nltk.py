from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import os

# txt_filename = 'covid19_abstract.txt'
# result_filename = 'abstract_split_nltk.txt'
txt_filename = 'covid19_title.txt'
result_filename = 'title_split_nltk.txt'
interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&',\
     '!', '*', '@', '#', '$', '%', '"', '``', "''", '<', '>', '=', '•',\
     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',\
     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '·', '\'',\
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '，', '-a', '‘', 'http',\
     '{', '}', '0123456789', '+', '-', '-0', '–', '”', '“', '\'s', '’', 'xx', 'ii']
stops = set(stopwords.words("english") + stopwords.words("french") +\
     stopwords.words("german") + stopwords.words("spanish") + stopwords.words("russian"))
lines_split = []

def tokenizer_nltk(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.lower()
        split_words_origin = word_tokenize(line)
        split_words_rmpunct = [word for word in split_words_origin if word not in interpunctuations]
        split_words_final = [word for word in split_words_rmpunct if word not in stops]
        words_tags = pos_tag(split_words_final)
        wnl = WordNetLemmatizer()
        split_words_lem = []
        for tag in words_tags:
            split_words_lem.append(wnl.lemmatize(tag[0], pos=wordnet.NOUN))
        lines_split.append(split_words_lem)
    pass

def write_result(lines, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            if len(line) > 0:
                for word in line:
                    f.write(word + ' ')  
                f.write('\n')
    pass



if __name__ == "__main__":
    tokenizer_nltk(txt_filename)
    write_result(lines_split, result_filename)

