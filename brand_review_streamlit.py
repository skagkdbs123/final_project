import streamlit as st
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import koreanize_matplotlib
import plotly.express as px

import koreanize_matplotlib
#%config InlineBackend.figure_format = 'retina'

from keybert import KeyBERT
from kiwipiepy import Kiwi

import requests
from bs4 import BeautifulSoup
import urllib.request
import cv2

st.set_page_config(
    page_title="brand review analysis",
    page_icon="👗",
    layout="wide",
)

st.markdown("# 👕 브랜드를 선택해주세요. 👖")

st.sidebar.markdown("# 브랜드 선택 ❓")

# select brand
brand_list = ['브랜드 선택', '라퍼지스토어', '꼼파뇨', '드로우핏', '인사일런스',
            '커버낫', '파르티멘토', '필루미네이트', '와릿이즌', '수아레',
            '내셔널지오그래픽', '예일', '디즈이스네버댓', '아웃스탠딩', '리',
            '어반드레스']
choice = st.selectbox('🔍보고자 하는 브랜드를 선택해주세요.', brand_list)

select_brand = []
for num in range(len(brand_list)):
    if num == 0:
        pass
    elif choice == brand_list[num]:
        st.write(f'{brand_list[num]}를 선택하셨군요. ⏳ 잠시만 기다려 주세요.')
        select_brand.append(f'{brand_list[num]}')
    

# data load
brand_link = {
    '라퍼지스토어' : '1fr6RZGM_vd5L0IDS0XIJCcScn_QtFKYz',
    '드로우핏' : '1KoWlv8cQXI9kpmfVL1pSr6jLH1RvkXkt',
    '커버낫' : '1we5q5975vDb2iNrmTPfDNsCNrQxrrWTQ',
    '파르티멘토' : '1D1BhygdkEvZU4uQQ1Y450zpZVHbdDiwc',
    '필루미네이트' : '190VQjL5F-8KPxQlYj3_8pY9Fb9JzmPIi',
    '꼼파뇨' : '1AjkzWni2Lp1V2vzhe3-gwzjDT1ASCsA2',
    '인사일런스' : '12vrFmoKeJ_UHaXljvsO1l4rbvV1Tw4Dq' ,
    '와릿이즌' : '1C_hPRBxb0sp6bJdVV4OEmNmMWFqYILQZ' ,
    '수아레' : '1KZSMUTjqqGGMVOp5KiH7X_n23IcJL3wB',
    '내셔널지오그래픽' : '1gWhGfIluszy8oVO-RwZIKtxJ0RM8yMhn',
    '예일' : '1eIoxQZrDLhsCWEl1-NJ9yE3Biem3VIcG',    
    '디즈이스네버댓' : '1ZgW4xBoXcNczxjMdw6YMKVEdjtgsUoyK',    
    '아웃스탠딩' : '1KfcwqNRRbKLgCNvgMIgZiRLaJ8Si8BJU',
    '리' : '1fqw2TiNEDxyrkaSDIlcxQ6UUyUD38kgW',
    '어반드레스' : '1YmNK_XSR03fcKgnt6ZOmWkAusXteV8tT' 
}

def data_load(select_brand):
    data_link = 'https://drive.google.com/uc?id='+brand_link[select_brand]
    data = pd.read_csv(data_link) 
    return data

  
try : 
    data_load_state = st.spinner('Loading data...') 
    data = data_load(select_brand[0])
except KeyError as k:
    pass
except IndexError as i:
    pass

# labeling
def labeling(data):
    df = data[["리뷰", "평점"]]
    df = df.reset_index(drop=True)

    삭제 = df[(df["평점"]=="60%") | (df["평점"]=="80%")].index
    df = df.drop(삭제)

    df.loc[(df["평점"] == "100%"), "label"] = 1
    df.loc[(df["평점"] == "20%") | (df["평점"] == "40%"), "label"] = 0

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# 문자 전처리
def preprocessing(text):
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA]', " ", text)
    text = re.sub('[\s]+', " ", text)
    text = text.lower()
    return text

try :
    label_data = labeling(data=data)
    positive = label_data[(label_data["평점"] == "100%")]
    negative = label_data[(label_data["평점"] == "20%") | (label_data["평점"] == "40%")]
    positive['리뷰'] = positive['리뷰'].map(preprocessing)
    negative['리뷰'] = negative['리뷰'].map(preprocessing)
    data_load_state.spinner(f'{select_brand[0]} 데이터 로드 success ‼')
except KeyError as k:
    pass
except NameError as n:
    pass

# Kiwi 적용
def kiwi(sentence):
    results = []
    result = Kiwi().analyze(sentence)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
            results.append(token)
    return results

try :
    for pos in positive['리뷰']:
        pos_noun = kiwi(sentence=pos)
    for neg in negative['리뷰']:
        neg_noun = kiwi(sentence=neg)
except KeyError as k:
    pass
except NameError as n:
    pass

# wordrank 적용
def keybert(df):
    array_text = pd.DataFrame(df["리뷰"]).to_numpy()

    keywords = []
    kw_extractor = KeyBERT('skt/kobert-base-v1')
    for j in range(len(array_text)):
        keyword = kw_extractor.extract_keywords(array_text[j][0])
        keywords.append(keyword)
    
    important = []
    for i in range(0, len(bow)):
        for j in range(len(bow[i])):
            important.append(bow[i][j])
            
    cum_count = pd.DataFrame([keywords, important], columns=['keyword', 'important'])

    keyword.groupby('keyword').agg('sum').sort_values('weight', ascending=False).head(200)
    #가중치 순서로 20개 시각화

# 호연 여기    

def keyword_review(select_brand, data, keywords):
    img_csv = pd.read_csv('https://drive.google.com/uc?id='+brand_link[select_brand])

    if st.button(keywords[0]):
        product_review = data[data['리뷰'].str.contains(f'{keywords[0]}')]['리뷰']
        product_num = data[data['리뷰'].str.contains(f'{keywords[0]}')]['상품_num']
