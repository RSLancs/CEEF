## python37

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
from lxml import etree

parser = ET.XMLParser(encoding="utf-8")


import csv
import re
from pprint import pprint
import pandas as pd
import glob
import pickle
import os

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_sm')

from pathlib import Path
from typing import List, Tuple, Set, Optional


##################################################################################
##.............extract all <rs> tags..............................................


def extract_tagged_elements(person, in_data: Path) -> List[Tuple[str, str]]:
    files_in_folder_list = glob.glob(f'{in_data}/*1867.xml')
    '''
    Parses the .xml folder that is passed to it. Extracts all 'rs' tag contents and re-structes the data into a dataframe
    '''
    print("Files which will be parsed, and 'rs' tags extracted : \n")

    tagged_elements = []
    for file in files_in_folder_list:
        print(file)

        xml_input = ET.parse(file) # use path provided to open .xml file
        
        for doc_date in xml_input.iter('date'):
            print(doc_date.attrib['when'])
            doc_date = doc_date.attrib['when']
        
        for rs in xml_input.iter('rs'):# collect as 'rs' tags in xml file
           

            if len(rs):# check rs tag for child tags and extract text is NOT gap
                for child in rs:
                    if child.tag == 'hi':
                        print('CHILD: ', child.text, rs.attrib, child.tag)
                        tagged_elements.append([person, doc_date, rs.attrib['type'],child.text])

                    if child.tag == 'unclear':
                     print('CHILD: ', child.text, rs.attrib, child.tag)
                     tagged_elements.append([person, doc_date, rs.attrib['type'],child.text])


            else:# if no child nodes extract text
                if rs.attrib['type'] != 'date':
                     print(rs.text, rs.attrib)
                     tagged_elements.append([person, doc_date, rs.attrib['type'],rs.text])

    my_df = pd.DataFrame(tagged_elements) # transform result list to dataframe
    my_df.columns = ['recipient ','letter date','element_type','element']

    return(my_df)

# Joan_tags = extract_tagged_elements('Joan','./Letters_to_Joan')

# margret_tags = extract_tagged_elements('Margret','./Letters_to_Margret')

##.............write tag elsms to csv.....................................
# outputs = pd.concat([Joan_tags, margret_tags])# merge tag sets from Joan and Margret
# outputs.to_csv('C:/Users/Rober/Box Sync/CEEF_work/Source set - Transcriptions/rs_tags_from_xml.csv', index=False, header=True)# write out csv



#####################################################################################
##.............transform xml text inside <p> tags into BIO.............................


def BIO(in_data: Path, output_path: Path) -> List[Tuple[str, str]]:

    files_in_folder_list = glob.glob(f'{in_data}/*1867.xml')
   
    print("Files which will be parsed for BIO reformatting : \n")

    print(f'BIO files to be written to {output_path}')

    for file in files_in_folder_list:

        xml_input = ET.parse(file) # use path provided to open .xml file
        root = xml_input.getroot()

        for text_id in root.iter('text'):#collect file id names for output save
            print(text_id.attrib['id'])
            text_id = text_id.attrib['id']
        
        for rs in root.iter('rs'):# collect as 'rs' tags in xml file
            attrib_type = str(rs.attrib['type'])
            tag = str(attrib_type).upper() + 'BIO'
            #print(type(tag),tag)

            if len(rs):# check rs tag for child tags and extract text is NOT gap
                for child in rs:
                    if child.tag == 'hi':
                        update_token = child.text + tag
                        child.text = str(update_token)
                        child.set('updated', 'yes')
                    
                    if child.tag == 'unclear':
                        update_token = child.text + tag
                        child.text = str(update_token)
                        child.set('updated', 'yes')
                        
            else:# if no child nodes extract text
                if rs.attrib['type'] != 'date':
                    update_token = rs.text + tag
                    rs.text = str(update_token)
                    rs.set('updated', 'yes')
                                 
        ## join whole senteces and add unique bio tag around each location 
        tagged_and_split = []
        
        for para in xml_input.iter('p'):    
            for sen in para:
                str_sen = tostring(sen)
                comp_sen_bio =''.join(ET.fromstring(str_sen).itertext())
                #print(comp_sen_bio)

                #...use spacy to tokenise words in sentence........
                nlp = spacy.load('en_core_web_sm')
                token_sen = nlp(comp_sen_bio)
                #print(token_sen)
                
                #........assign BIO location tags to tokens.......
                for ent in token_sen:
                    #print(ent.text)      

                    ## mark up all person tags 
                    if 'PERSONBIO' in ent.text:
                        text_ent = ent.text
                        ent_splits = [e.strip() for e in text_ent.split('PERSONBIO') if e != '']
                        #print(ent_splits)
                        if len(ent_splits) == 1:
                            tagged_and_split.append(ent_splits[0] +' B-PER')
                        elif len(ent_splits) == 2:
                            tagged_and_split.append(ent_splits[0] +' B-PER')
                            tagged_and_split.append(ent_splits[1] +' I-PER')
                        elif len(ent_splits) >= 3:
                            tagged_and_split.append(ent_splits[0] +' B-PER')
                            for i in ent_splits[1:]:
                                tagged_and_split.append(i +' I-PER')               
                    
                    ## mark up all place tags 
                    elif 'PLACEBIO' in ent.text:
                        text_ent = ent.text
                        ent_splits = [e.strip() for e in text_ent.split('PLACEBIO') if e != '']
                        #print(ent_splits)
                        if len(ent_splits) == 1:
                            tagged_and_split.append(ent_splits[0] +' B-LOC')
                        elif len(ent_splits) == 2:
                            tagged_and_split.append(ent_splits[0] +' B-LOC')
                            tagged_and_split.append(ent_splits[1] +' I-LOC')
                        elif len(ent_splits) >= 3:
                            tagged_and_split.append(ent_splits[0] +' B-LOC')
                            for i in ent_splits[1:]:
                                tagged_and_split.append(i +' I-LOC')               
                    
                    ## mark up all flora tags 
                    elif 'FLORABIO' in ent.text:
                        text_ent = ent.text
                        ent_splits = [e.strip() for e in text_ent.split('FLORABIO') if e != '']
                        #print(ent_splits)
                        if len(ent_splits) == 1:
                            tagged_and_split.append(ent_splits[0] +' B-FLO')
                        elif len(ent_splits) == 2:
                            tagged_and_split.append(ent_splits[0] +' B-FLO')
                            tagged_and_split.append(ent_splits[1] +' I-FLO')
                        elif len(ent_splits) >= 3:
                            tagged_and_split.append(ent_splits[0] +' B-FLO')
                            for i in ent_splits[1:]:
                                tagged_and_split.append(i +' I-FLO')

                    ## mark up all fauna tags 
                    elif 'FAUNABIO' in ent.text:
                        text_ent = ent.text
                        ent_splits = [e.strip() for e in text_ent.split('FAUNABIO') if e != '']
                        #print(ent_splits)
                        if len(ent_splits) == 1:
                            tagged_and_split.append(ent_splits[0] +' B-FAU')
                        elif len(ent_splits) == 2:
                            tagged_and_split.append(ent_splits[0] +' B-FAU')
                            tagged_and_split.append(ent_splits[1] +' I-FAU')
                        elif len(ent_splits) >= 3:
                            tagged_and_split.append(ent_splits[0] +' B-FAU')
                            for i in ent_splits[1:]:
                                tagged_and_split.append(i +' I-FAU')

                    ## mark up all mineral tags 
                    elif 'MINERALBIO' in ent.text:
                        text_ent = ent.text
                        ent_splits = [e.strip() for e in text_ent.split('MINERALBIO') if e != '']
                        #print(ent_splits)
                        if len(ent_splits) == 1:
                            tagged_and_split.append(ent_splits[0] +' B-MIN')
                        elif len(ent_splits) == 2:
                            tagged_and_split.append(ent_splits[0] +' B-MIN')
                            tagged_and_split.append(ent_splits[1] +' I-FAU')
                        elif len(ent_splits) >= 3:
                            tagged_and_split.append(ent_splits[0] +' B-MIN')
                            for i in ent_splits[1:]:
                                tagged_and_split.append(i +' I-MIN')

                    else:
                        if ent.text != ' ':
                            tagged_and_split.append(ent.text + ' O')  
                             
                tagged_and_split.append('')

        #............write out BIO file..................
        out_bio_file = open(output_path + str(text_id) + '_BIO'+ '.txt', 'w', encoding='utf8')
        
        for item in tagged_and_split:
            out_bio_file.write("%s\n" % item) 
   
    #return tagged_and_split


#margret_text = BIO('./Letters_to_Margret/','./Margret_letters_BIO/')

#joan_text = BIO('./Letters_to_Joan/','./Joan_letters_BIO/')


#############################################################################
##.........extract raw text inside <p> tags................................


def raw_text(in_data: Path, output_path: Path) -> List[Tuple[str, str]]:
    files_in_folder_list = glob.glob(f'{in_data}/*1867_BIO.txt')
   
    print(f'BIO files to be written to {output_path} \n')
    
    print("Files where raw text is extracted : \n")
    
    for file in files_in_folder_list:
        #print(file)
        
        file_name_in = os.path.basename(file)# collect file name
        file_name_out = re.sub(r'_BIO.txt', r'',file_name_in) # remove 'BIO' extension
        print(file_name_out)

        with open(file, 'r') as f: 
            text_item = f.read().split('\n')

        just_text = []
        for word in text_item:
            # print(len(word), word)
            
            if len(word) >= 1:
                splits = word.split()
                just_text.append(splits[0])
                
        raw_text =  ' '.join(just_text)       
        #print(raw_text)
        
        ##............write raw text to file..................
        out_raw_file = open(output_path + file_name_out + '_RAW.txt', 'w', encoding='utf8')
        out_raw_file.write(raw_text)

    #return raw_text


#joan_text = raw_text('./Joan_letters_BIO/','./Joan_letters_raw_text/')

#margret_text = raw_text('./Margret_letters_BIO/','./Margret_letters_raw_text/')




