# final_project

## 데이터에 관하여
[MUSINSA(클릭하면 무신사 홈페이지로 이동)](https://www.musinsa.com/app/)

| 데이터 출처  |  무신사 상품 정보 및 리뷰 데이터 크롤링 |
| --- | --- |
| 데이터 수집기준 | 각 브랜드 별  상품에서  별점 낮은순  900개씩 데이터수집(낮은리뷰 갯수가 적어서 부정리뷰외에도 긍정리뷰 다수포함) |
| 수집된 브랜드 | 라퍼지스토어, 꼼파뇨, 드로우핏, 인사일런스, 커버낫, 파르티멘토, 필루미네이트, 와릿이즌, 수아레, 내셔널지오그래픽, 예일, 디즈이스네버댓, 아웃스탠딩, 리, 어반드레스 |
| 수집한 내용  | 상품명 / 상품의 리뷰 /  평점 /  제품사이즈 / (리뷰를 작성한) 닉네임, 레벨/ 카테고리/상품 넘버  |

## 예상 시안

![예상시안1](/img_folder/결과물예상도1.png)
![예상시안2](/img_folder/결과물예상도2.png)

실제 결과물
![결과물1](/img_folder/결과물1.png)
![결과물2](/img_folder/결과물2.png)
![결과물3](/img_folder/결과물3.png)


## 키워드 추출에 사용한 라이브러리

1. keybert(Tokenizer를 통한 키워드 추출)
2. okt,kiwi(명사만 따로 자르기 위한 형태소 분석기)
3. Krwordrank(Tokenizer를 사용하지 않는 키워드 추출)
