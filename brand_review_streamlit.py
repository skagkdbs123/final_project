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

from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank
from kiwipiepy import Kiwi

import requests
from bs4 import BeautifulSoup


st.set_page_config(
    page_title="brand review analysis",
    page_icon="๐",
    layout="wide",
)

st.markdown("# ๐ ๋ธ๋๋๋ฅผ ์ ํํด์ฃผ์ธ์. ๐")

st.sidebar.markdown("# ๐ ๋ธ๋๋ ์ข๋ฅ")
st.sidebar.markdown("""
๋ผํผ์ง์คํ ์ด lafudgestore            
๊ผผํ๋จ compagno         
๋๋ก์ฐํ Draw fit         
์ธ์ฌ์ผ๋ฐ์ค insilence       
์ปค๋ฒ๋ซ covernat         
ํ๋ฅดํฐ๋ฉํ  partimento         
ํ๋ฃจ๋ฏธ๋ค์ดํธ filluminate       
์๋ฆฟ์ด์ฆ whatitisnt       
์์๋  suare        
๋ด์๋์ง์ค๊ทธ๋ํฝ nationalgeographic           
์์ผ yale       
๋์ฆ์ด์ค๋ค๋ฒ๋ thisisneverthat         
์์์คํ ๋ฉ outstanding        
๋ฆฌ lee      
์ด๋ฐ๋๋ ์ค avan         
""")

# select brand
brand_list = ['๋ธ๋๋ ์ ํ', '๋ผํผ์ง์คํ ์ด', '๊ผผํ๋จ', '๋๋ก์ฐํ', '์ธ์ฌ์ผ๋ฐ์ค',
            '์ปค๋ฒ๋ซ', 'ํ๋ฅดํฐ๋ฉํ ', 'ํ๋ฃจ๋ฏธ๋ค์ดํธ', '์๋ฆฟ์ด์ฆ', '์์๋ ',
            '๋ด์๋์ง์ค๊ทธ๋ํฝ', '์์ผ', '๋์ฆ์ด์ค๋ค๋ฒ๋', '์์์คํ ๋ฉ', '๋ฆฌ',
            '์ด๋ฐ๋๋ ์ค']
choice = st.selectbox('๐๋ณด๊ณ ์ ํ๋ ๋ธ๋๋๋ฅผ ์ ํํด์ฃผ์ธ์.', brand_list)

select_brand = []
for num in range(len(brand_list)):
    if num == 0:
        pass
    elif choice == brand_list[num]:
        st.write(f'{brand_list[num]}๋ฅผ ์ ํํ์จ๊ตฐ์. โณ ์ ์๋ง ๊ธฐ๋ค๋ ค ์ฃผ์ธ์.')
        select_brand.append(f'{brand_list[num]}')
    

# data load
brand_link = {
    '๋ผํผ์ง์คํ ์ด' : '1fr6RZGM_vd5L0IDS0XIJCcScn_QtFKYz',
    '๋๋ก์ฐํ' : '1KoWlv8cQXI9kpmfVL1pSr6jLH1RvkXkt',
    '์ปค๋ฒ๋ซ' : '1we5q5975vDb2iNrmTPfDNsCNrQxrrWTQ',
    'ํ๋ฅดํฐ๋ฉํ ' : '1D1BhygdkEvZU4uQQ1Y450zpZVHbdDiwc',
    'ํ๋ฃจ๋ฏธ๋ค์ดํธ' : '190VQjL5F-8KPxQlYj3_8pY9Fb9JzmPIi',
    '๊ผผํ๋จ' : '1AjkzWni2Lp1V2vzhe3-gwzjDT1ASCsA2',
    '์ธ์ฌ์ผ๋ฐ์ค' : '12vrFmoKeJ_UHaXljvsO1l4rbvV1Tw4Dq' ,
    '์๋ฆฟ์ด์ฆ' : '1C_hPRBxb0sp6bJdVV4OEmNmMWFqYILQZ' ,
    '์์๋ ' : '1KZSMUTjqqGGMVOp5KiH7X_n23IcJL3wB',
    '๋ด์๋์ง์ค๊ทธ๋ํฝ' : '1gWhGfIluszy8oVO-RwZIKtxJ0RM8yMhn',
    '์์ผ' : '1eIoxQZrDLhsCWEl1-NJ9yE3Biem3VIcG',    
    '๋์ฆ์ด์ค๋ค๋ฒ๋' : '1ZgW4xBoXcNczxjMdw6YMKVEdjtgsUoyK',    
    '์์์คํ ๋ฉ' : '1KfcwqNRRbKLgCNvgMIgZiRLaJ8Si8BJU',
    '๋ฆฌ' : '1fqw2TiNEDxyrkaSDIlcxQ6UUyUD38kgW',
    '์ด๋ฐ๋๋ ์ค' : '1YmNK_XSR03fcKgnt6ZOmWkAusXteV8tT' 
}

@st.cache
def data_load(select_brand):
    data_link = 'https://drive.google.com/uc?id='+brand_link[select_brand]
    data = pd.read_csv(data_link) 
    return data

def graph(brand):
    ์ญ์  = brand[(brand["ํ์ "]=="60%") | (brand["ํ์ "]=="80%")].index
    brand = brand.drop(์ญ์ )
    brand.loc[(brand["ํ์ "] == "100%"), "label"] = 1
    brand.loc[(brand["ํ์ "] == "20%") | (brand["ํ์ "] == "40%"), "label"] = 0
    brand = brand.drop_duplicates()
    brand = brand.reset_index(drop=True)
    brand["ํ์ "].value_counts()

    dfbrand๊ธ์  = brand[brand["label"]==1]
    dfbrand๊ธ์  = dfbrand๊ธ์ .reset_index(drop=True)
    dfbrand๋ถ์  = brand[brand["label"]==0]
    dfbrand๋ถ์  = dfbrand๋ถ์ .reset_index(drop=True)

    dfbrand์ฌ์ด์ฆ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('์ฌ์ด์ฆ')]
    branda = dfbrand์ฌ์ด์ฆ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand์๊ฐ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('์๊ฐ')]
    brandb = dfbrand์๊ฐ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand์ฌ์ง = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('์ฌ์ง')]
    brandc = dfbrand์ฌ์ง.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๋๋ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๋๋')]
    brandd = dfbrand๋๋.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๋์์ธ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๋์์ธ')]
    brande = dfbrand๋์์ธ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrandํ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('์ค๋ฒ|๋ฒํ')]
    brandf = dfbrandํ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๋๊ป = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๋๊ป')]
    brandg = dfbrand๋๊ป.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand์์ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('์์')]
    brandh = dfbrand์์.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๊ฐ๊ฒฉ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๊ฐ๊ฒฉ')]
    brandi = dfbrand๊ฐ๊ฒฉ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๊ธฐ์ฅ = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๊ธฐ์ฅ|๊ธธ์ด')]
    brandj = dfbrand๊ธฐ์ฅ.shape[0] / dfbrand๊ธ์ .shape[0]

    dfbrand๋ก๊ณ  = dfbrand๊ธ์ [dfbrand๊ธ์ ['๋ฆฌ๋ทฐ'].str.contains('๋ก๊ณ ')]
    brandk = dfbrand๋ก๊ณ .shape[0] / dfbrand๊ธ์ .shape[0]


    dfbrand๋ฐฐ์ก = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๋ฐฐ์ก')]
    brandl = dfbrand๋ฐฐ์ก.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand์ฌ์ด์ฆ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('์ฌ์ด์ฆ')]
    brandm = dfbrand์ฌ์ด์ฆ.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand์ธํ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('์ธํ')]
    brandn = dfbrand์ธํ.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrandํ์ง = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('ํ์ง')]
    brando = dfbrandํ์ง.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๊ตํ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๊ตํ')]
    brandp = dfbrand๊ตํ.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand์ฌ์ง = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('์ฌ์ง')]
    brandq = dfbrand์ฌ์ง.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๊ฐ๊ฒฉ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๊ฐ๊ฒฉ')]
    brandr = dfbrand๊ฐ๊ฒฉ.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๋๋ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๋๋')]
    brands = dfbrand๋๋.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๋์ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๋์')]
    brandt = dfbrand๋์.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๋ณดํ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๋ณดํ')]
    brandu = dfbrand๋ณดํ.shape[0] / dfbrand๋ถ์ .shape[0]

    dfbrand๋ง๊ฐ = dfbrand๋ถ์ [dfbrand๋ถ์ ['๋ฆฌ๋ทฐ'].str.contains('๋ง๊ฐ')]
    brandv = dfbrand๋ง๊ฐ.shape[0] / dfbrand๋ถ์ .shape[0]

    listbrand๊ธ์ = [branda, brandb, brandc, brandd, brande, brandf, brandg, brandh, brandi, brandj, brandk]
    listbrand๋ถ์ = [brandl, brandm, brandn, brando, brandp, brandq, brandr, brands, brandt, brandu, brandv]

    Series3 = pd.Series(listbrand๊ธ์ )
    Series4 = pd.Series(listbrand๋ถ์ )

    dfbrand๊ธ์ ๋น์จ = pd.DataFrame({"ํค์๋": Series1, "๋น์จ": Series3})

    dfbrand๋ถ์ ๋น์จ = pd.DataFrame({"ํค์๋": Series2, "๋น์จ": Series4})

    # import matplotlib.pyplot as plt

    st.line_chart(word๊ธ์ , list๊ธ์ , title='๊ธ์  ํค์๋ ๋น๊ต')

    st.line_chart(word๊ธ์ , listbrand๊ธ์ , title = '๊ธ์  ํค์๋ ๋น๊ต')

    st.imshow()

    st.line_chart(word๋ถ์ , list๋ถ์ , title='๋ถ์  ํค์๋ ๋น๊ต')

    st.line_chart(word๋ถ์ , listbrand๋ถ์ , title='๋ถ์  ํค์๋ ๋น๊ต')

    st.imshow()
  
try : 
    data_load_state = st.text('Loading data...') 
    data = data_load(select_brand[0])
    graph(data)
    pos_data = data[data["ํ์ "] == "100%"]
    neg_data = data[(data["ํ์ "] == "20%") | (data["ํ์ "] == "40%")]
except KeyError as k:
    pass
except IndexError as i:
    pass
except NameError as n:
    pass
# labeling
def labeling(data):
    df = data[["๋ฆฌ๋ทฐ", "ํ์ "]]
    df = df.reset_index(drop=True)

    ์ญ์  = df[(df["ํ์ "]=="60%") | (df["ํ์ "]=="80%")].index
    df = df.drop(์ญ์ )

    df.loc[(df["ํ์ "] == "100%"), "label"] = 1
    df.loc[(df["ํ์ "] == "20%") | (df["ํ์ "] == "40%"), "label"] = 0

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# ๋ฌธ์ ์ ์ฒ๋ฆฌ
def preprocessing(text):
    text = re.sub('[^๊ฐ-ํฃใฑ-ใใ-ใฃa-zA]', " ", text)
    text = re.sub('[\s]+', " ", text)
    text = text.lower()
    return text

try :
    label_data = labeling(data=data)
    positive = label_data[(label_data["ํ์ "] == "100%")].sample(10)
    negative = label_data[(label_data["ํ์ "] == "20%") | (label_data["ํ์ "] == "40%")].sample(10)
    positive['๋ฆฌ๋ทฐ'] = positive['๋ฆฌ๋ทฐ'].map(preprocessing)
    negative['๋ฆฌ๋ทฐ'] = negative['๋ฆฌ๋ทฐ'].map(preprocessing)
except KeyError as k:
    pass
except NameError as n:
    pass

# Kiwi ์ ์ฉ
def noun_extractor(sentence):
    results = []
    result = Kiwi().analyze(sentence)
    for token, pos, _, _ in result[0][0]:
        if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
            results.append(token)
    return results

try :
    pos_noun_list = []
    neg_noun_list = []
    print('๊ธ์  ๋ฆฌ๋ทฐ kiwi')
    for pos in positive['๋ฆฌ๋ทฐ'].tolist():
        pos_nouns = noun_extractor(pos)
        pos_text = ' '.join(pos_nouns)
        pos_noun_list.append(pos_text)

    print('๋ถ์  ๋ฆฌ๋ทฐ kiwi')
    for neg in negative['๋ฆฌ๋ทฐ'].tolist():
        neg_nouns = noun_extractor(neg)
        neg_text = ' '.join(neg_nouns)
        neg_noun_list.append(neg_text)

except KeyError as k:
    pass
except NameError as n:
    pass

# wordrank ์ ์ฉ top10 ํค์๋ ๋ฝ๊ธฐ
def word_rank(corpus):
    beta = 0.90    # PageRank์ decaying factor beta
    max_iter = 5
    top_keywords = []
    fnames = [corpus]
    top_10=[]
    
    for fname in fnames:
        texts = fname
        wordrank_extractor = KRWordRank(min_count=1, max_length=10, verbose=False)
        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
        top_keywords.append(sorted(keywords.items(),key=lambda x:x[1],reverse=True)[:100])
        
    for i in range(len(top_keywords)):
        if i<10:
            top_10.append(top_keywords[0][i][0])
            i += 1
    return top_10

try :
    pos_keyword = word_rank(pos_noun_list)
    neg_keyword = word_rank(neg_noun_list)
except KeyError as k:
    pass
except NameError as n:
    pass
except ValueError as v:
    pass

# img data load
img_brand_link = {
    '๋ผํผ์ง์คํ ์ด' : '1QfwKjzDAfoowpe4LiH9yxhChdZo3iizh',
    '๋๋ก์ฐํ' : '1-caqKnBlM4Q26tec_4aWFm9017F9-_E4',
    '์ปค๋ฒ๋ซ' : '1f50QDJI6K7KeZ7WGSANV1sH6aiXa2c5T',
    'ํ๋ฅดํฐ๋ฉํ ' : '1Nt0LAAWlvVTh60Y9Zvbb3jmw0Jdl-URg',
    'ํ๋ฃจ๋ฏธ๋ค์ดํธ' : '1CtYGt4E5hzqp-tvwix5WhANpvzds8TQK',
    '๊ผผํ๋จ' : '1-CJcWAp3WxKymk7PUxj9NYgM-3zfjM83',
    '์ธ์ฌ์ผ๋ฐ์ค' : '1COUpes3WPXeGn6mdYrtzG1D9dMJ43SF7' ,
    '์๋ฆฟ์ด์ฆ' : '1nxOa5_69KQrmbMduDhQtAPFgiclIFXO_' ,
    '์์๋ ' : '1G5QtrNYtKNhFgRVx0Fj2-b0F626o7ccX',
    '๋ด์๋์ง์ค๊ทธ๋ํฝ' : '1Wm2ox9koFbYXkMtvLApCfQjDnPfSRsGF',
    '์์ผ' : '1MxqLCSCptl5O5shldxK513mh4G7z3j3V',    
    '๋์ฆ์ด์ค๋ค๋ฒ๋' : '17yuM4U3W3aKKMCQphvBePiHC5mVugVkf',    
    '์์์คํ ๋ฉ' : '17v-GwoTF0mgOkRta3hAhytWLPOh2YUpk',
    '๋ฆฌ' : '1us6tb40vHoz4hrNGfoD9n2BL0fciY97n',
    '์ด๋ฐ๋๋ ์ค' : '1LDAyqzM-TZZCLVrZJK08WLJe2IF6Jtxt' 
}

def img_data_load(select_brand):
    for brand in select_brand:
        img_data_link = f'https://drive.google.com/uc?id='+img_brand_link[brand]
        img_csv = pd.read_csv(img_data_link) 
    return img_csv  


def keyword_review(img_link, df, keywords):
    # ๊ฐ ํค์๋ for๋ฌธ์ผ๋ก ๋๋ฆฌ๊ธฐ
    for key_count in range(len(keywords)):
        # ํค์๋์ ๋จ์ด๋ฅผ ํฌํจํ๋ ๋ฆฌ๋ทฐ๋ฅผ keyword_review_data๋ก ํ ๋น
        keyword_review_data = df[df['๋ฆฌ๋ทฐ'].str.contains(keywords[key_count])]

        # ๊ทธ ํ ๋นํ ๋ณ์์์ ์ํ ๋ฒํธ๋ฅผ ๊ฐ์ ธ์ค๊ณ , ๊ฐ์ฅ ๋ง์ด ์ฐจ์งํ๋ top3์ ์ํ ๋ฒํธ๋ฅผ ๊ฐ์ ธ์ค๊ธฐ
        product_num = keyword_review_data['์ํ_num']
        top3_cumc_product = product_num.value_counts().sort_values(ascending=False)[:3].index

        # top3์ ์ํ ๋ฒํธ์ ํด๋นํ๋ ๋ฆฌ๋ทฐ ๊ฐ๊ฐ 3๊ฐ์ ๋ฆฌ๋ทฐ ๊ฐ์ ธ์ค๊ธฐ
        review1 = keyword_review_data[keyword_review_data['์ํ_num']==top3_cumc_product[0]]['๋ฆฌ๋ทฐ'].sample(3)
        review2 = keyword_review_data[keyword_review_data['์ํ_num']==top3_cumc_product[1]]['๋ฆฌ๋ทฐ'].sample(3)
        review3 = keyword_review_data[keyword_review_data['์ํ_num']==top3_cumc_product[2]]['๋ฆฌ๋ทฐ'].sample(3)

        st.button(keywords[key_count])
        
        for i, number in enumerate(top3_cumc_product):
            # ์ํ๋ช
            product_name_list = keyword_review_data[keyword_review_data['์ํ_num'] == top3_cumc_product[i]]['์ํ']
            product_name = list(set(product_name_list))
            st.text(f'์ํ ์ต์ : {product_name}')

            # ์ด๋ฏธ์ง ๋งํฌ
            #imgs_link = img_link[img_link['์ํ_num'] == number]['์ฌ์ง'].values
            #join_link = ''.join(imgs_link)
            #link = f'https:{join_link}'
            #st.text(f'์ด๋ฏธ์ง ๋งํฌ : {link}')
            
            #image = Image.open(f'{img_link}')
            #st.image(image)

            if i==0:
                st.text(review1.values)
            if i==1:
                st.text(review2.values)
            if i==2:
                st.text(review3.values)

#            link = img_link[img_link['์ํ_num'] == num]['์ฌ์ง'].values
#            join_link = ''.join(link)
#            URL = f'https:{join_link}'
#            response = requests.get(URL)
#            image = Image.open(BytesIO(response.content))
#            st.image(image, caption='Image from URL')
        

try :
    link_csv = img_data_load(select_brand)
    st.markdown("""## ๐ ๊ธ์  ๋ฆฌ๋ทฐ ํต์ฌ ํค์๋""")
    pos = keyword_review(link_csv, pos_data, pos_keyword)
    st.text(pos)
    st.markdown("""## ๐ ๋ถ์  ๋ฆฌ๋ทฐ ํต์ฌ ํค์๋""")
    neg = keyword_review(link_csv, neg_data, neg_keyword)
    st.text(neg)
except KeyError as k:
    pass
except IndexError as i:
    pass
except NameError as n:
    pass


