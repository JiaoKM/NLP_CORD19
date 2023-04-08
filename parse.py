import json
import os

path_1 = 'document_parses\pdf_json'
path_2 = 'document_parses\pmc_json'
# txt_filename = 'covid19_paper.txt'
# txt_filename = 'covid19_abstract.txt'
txt_filename = 'covid19_title.txt'
text = []

def getfiles(path):
    filenames = os.listdir(path)
    return filenames

def read_json(path, filenames):
    for i in range(0, len(filenames)):
        file = path + '\\' + filenames[i]
        with open(file) as f:
            paper = json.load(f)
        text.append(paper['metadata']['title'])
        '''
        if path == path_1:
            if len(paper['abstract']) != 0:
                for j in range(0, len(paper['abstract'])):
                    text.append(paper['abstract'][j]['text'])
        '''
        
        '''
        for j in range(0, len(paper['body_text'])):
            text.append(paper['body_text'][j]['text'])
        '''
        

def write_txt(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(0, len(text)):
            f.write(text[i] + '\n')



if __name__ == "__main__":
    filenames_1 = getfiles(path_1)
    filenames_2 = getfiles(path_2)
    read_json(path_1, filenames_1)
    read_json(path_2, filenames_2)
    write_txt(txt_filename, text)

