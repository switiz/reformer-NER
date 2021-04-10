# Reformer - NER Subtask
[Reformer Language Model](https://github.com/nawnoes/reformer-language-model)

Seonghwan Kim님이 공유한 Reformer 모델 구현체 중 MLM모델로 Pre-trained하여 NER Task를 수행

Reformer에 대한 자세한 내용에 대한 정리는 Seonghwan Kim님 [Blog](https://velog.io/@nawnoes/Reformer-%EA%B0%9C%EC%9A%94) 에 잘 정리가 되어있습니다.

---
## Reformer
2020년에 발표 된 모델로, `LSH(Local Sensitive Hashing)`, `RevNet(Reversivle Residual Network)`, `Chunked Feed Forward Layer`,
`Axial Positional Encodings`을 통해서 트랜스포머의 메모리 문제를 개선하고자 시도한 모델. 

### Transformer의 단점
- `attention 계산`: 길이 **L**을 가진 문장의 어텐션을 계산할 때, **O(L^2)** 의 메모리와 시간 복잡도를 가진다.
- `많은 수의 레이어`: **N**개의 레이어틑 **N**배의 많은 메모리를 사용한다. 그리고 각각의 레이어는 역전파 계산을 위해 그 값들을 저장해둔다.
- `Feed Forward 레이어의 크기`: Feed Forward 레이어가 Attention의 Activation 깊이 보다 더 클 수 있다. 
### Reformer
- `LSH(Local Sensitive Hashing)`: Dot-Product 사용하는 기존의 어텐션을 locality-sensitive hashing을 사용해 대체하면 기존의 O(L^2)을
O(L log(L))로 개선
- `RevNet`: 트랜스포머에서는 Residual Network에서 backpropagation를 위해 gradient 값을 저장하고 있다. reversible residual network을 이용하여
메모리 문제를 계산 문제로 바꾸어 메모리를 문제를 개선
- `Chunk`: Feed Forward layer의 각 부분은 위치와 관계 없이 독립적이기 때문에 청크 단위로 잘라서 계산할 수 있다. 이 점을 이용하여 메모리에 올릴 때 청크 단위로 메모리에 올려, 메모리 효율을 개선. 
- `Axial Positional Encoding`: 매우 큰 input sequence에 대해서도 positional encoding을 사용할 수 있게 하는 방법. 


### Vocab & Tokenizer
`Sentencepiece`와 `Wordpiece` 중 기존에 사용해보지 않은 Wordpiece Tokenizer 사용.

### Data 
#### 사용 데이터
- 한국어 위키: [ratsgo님 전처리 데이터](https://ratsgo.github.io/embedding/downloaddata.html)

### GPU
-  RTX 3080 (epoch당 1hr 30m)

### Language Model - Masked Language Model(ex. BERT) 
BERT에서 사용한 Masked Language Model을 이용한 언어모델 학습. NSP와 SOP 없이 학습.
![](./images/mlm.png)
#### Model
##### BERT Model Config
|   |H=128|H=256|H=512|H=768|
|---|:---:|:---:|:---:|:---:|
| **L=2**  |[**2/128 (BERT-Tiny)**]|[2/256]|[2_512]|[2_768]|
| **L=4**  |[4/128]|[**4/256 (BERT-Mini)**]|[**4/512 (BERT-Small)**]|[4/768]|
| **L=6**  |[6/128]|[6/256]|[6/512]|[6/768]|
| **L=8**  |[8/128]|[8/256]|[**8/512 (BERT-Medium)**]|[8/768]|
| **L=10** |[10/128]|[10/256]|[10/512]|[10/768]|
| **L=12** |[12/128]|[12/256]|[12/512]|[**12/768 (BERT-Base)**]|
##### mlm-pretrain-small
**BERT-base**의 1/3 크기의 레이어 사용.
```
max_len = 512   
batch_size = 64
dim = 512
depth = 4
heads = 8
causal = False
```
#### 학습 데이터
한국어 위키 512 토큰 이하로 나누어 데이터 생성
```
[CLS] 구명은 분구될 때, 이 지역의 명산인 관악산(冠岳山)에서 따왔다.관악구의 행정 구역은 봉천동, 신림동, 남현동 3개의 법정동을 21개의 행정동으로 관리를 하고 있다. [SEP] 관악구의 면적은 29.57km이며, 인구는 2012년 12월 31일을 기준으로 247,598세대, 540,520명이다. [SEP] 서울 지하철 4호선 사당역은 서초구 및 동작구의 경계상에 접하고 있어 사실상 관악구의 전철역으로 보아도 무방하다. [SEP] 서울 지하철 7호선은 청림동을 통과하지만, 정거장은 없다. [SEP]  [SEP] 서초구 [SEP] 서초구(瑞草區)는 대한민국 서울특별시 남동부에 있는 구이다. [SEP] 남쪽으로는 경기도 성남시와 과천시, 동쪽으로는 강남구, 서쪽으로는 동작구와 관악구, 북쪽으로는 한강을 경계로 용산구와 접한다. [SEP] 면적은 약 47km로 서울특별시에서 가장 넓다. [SEP] 1988년에 강남구에서 분리되었다. [SEP] 구명은 분구될 때, 이 지역의 중심지인 서초동(瑞草洞)에서 유래하며, 서초동의 지명은 옛날 이곳이 서리풀이 무성했다 [SEP] 하여 이름붙여진 상초리(霜草里) 혹은 서초리(瑞草里)에서 온 것이다. [SEP] 또는 물이 서리어 흐르는 벌판이란 뜻으로 '서릿벌'이라 불렸는데 이것이 변해 서리풀이 된 것이라고도 한다. [SEP] 서초구는 서울특별시 한강 이남의 중앙부에 위치한 지역으로 동쪽으로는 강남구, 서쪽으로는 동작구와 관악구, 남쪽으로는 경기도 과천시와 성남시가 인접하고 북쪽으로는 한강 건너 용산구와 마주하고 있다. [SEP] 강남대로를 경계로 강남구, 동작대로 및 현충로, 남부순환로를 경계로 동작구, 관악구, 반포대교 및 한남대교 등을 경계로 용산구, 남태령을 경계로 경기도 과천시와 경부고속도로 및 청계산을 경계로 경기도 성남시와 경계를 이루고 있다. [SEP] 서초구의 행정 구역은 10개의 법정동을 18개의 행정동이 관리를 하고 있다. [SEP] 서초구의 면적은 47.00km이며, 서울시의 7.8%를 차지하고 있다. [SEP] 서초구의 인구는 2018년 2분기를 기준으로 443,989명, 174,268세대이다. [SEP] 
[CLS] 조선 제3대 태종과 그의 비 원경왕후의 묘인 헌릉과 제23대 순조와 그의 비 순원왕후 합장되어 있는 묘인 인릉이 있다.둘을 합하여 헌인릉이라 한다. [SEP] (내곡동 소재) [SEP] 구룡산(九龍山 : 306m)기슭에 세종대왕릉(英陵)이 있었으나, 영릉은 1469년(예종 1년)에 여주로 천장(遷葬)하였다. [SEP] (내곡동 소재) [SEP] 예술의 전당은 대한민국 최대 규모의 종합예술시설로서, 오페라하우스, 미술관, 서예관, 음악관 등이 있고, 예술의 전당 바로 옆에는 국립국악기관인 국립국악원이 있다. [SEP] (서초동 소재) [SEP] 서초동에는 교보문고, 대법원, 검찰청, 국립중앙도서관등이 있다. [SEP] 반포4동에는 신세계백화점, 마르퀘스플라자, 메리어트호텔, 호남선 고속버스터미널 등이 있는 센트럴시티가 있고 바로 옆에는 경부선 등 고속버스터미널이 있는 서울고속버스터미널이 있다. [SEP] 반포한강공원에는 세빛섬과 반포대교 달빛무지개 분수 등의 명소가 있다. [SEP] 서초구 양재동 일대에 자리잡고 있는 양재시민의 숲은 가족단위로 나들이하기에 좋은 공원으로 1986년 개장한 곳이다. 가을의 은행나무 낙엽길이 유명하다. [SEP] 양재동에는 유스호스텔인 서울교육문화회관과, 반포동 고속터미널에 JW 메리어트호텔 서울, 서울팔레스호텔이 위치하고 있다. [SEP] 방배동은 먹자골목으로 유명하다. [SEP] 1976년부터 한자리를 지켜온 레스토랑 "장미의 숲"은 1980년대 방배동을 "최고의 카페촌"으로 부상케 한 대표적인 곳이다. [SEP] 카페 "밤과 음악사이"는 1970~80년대 가요 [SEP] 와 인테리어를 하고 있으며, 통골뱅이와 김치찌개가 대표적인 안줏거리이다. [SEP] "멋쟁이 카페"들로 명성을 날렸던 방배동은 청담동에 그 명성을 내준 대신 요즘은 먹자골목으로 유명하다. [SEP] 이 일대는 아귀찜을 전문으로 하는 식당이 많다. [SEP] 
```

#### Pretraining 결과

10 Epochs Train Eval Losses : 2.3536606443871695

---
## NER subtask Fine-Tuning

PreTrain Reformer를 LM으로 이용하고 NER classifier(FFN) 로 Classification 함.  

### 학습데이터 
Naver NER dataset : [github link](https://github.com/naver/nlp-challenge)
- 해당 데이터셋에 Train dataset만 존재하기에, monologg님이 kobert-ner에 공유한 split된 data를 이용하였습니다.
[(Data link)](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging/data)

  Train (81,000) / Test (9,000)

### 모델설정
ner-pretrain-small.json
```t
reformer + NER classifier
```

### 학습결과

- 10 epochs Fine-Turning 진행함. (epoch당 15m)

```
              precision    recall  f1-score   support

         AFW       0.00      0.00      0.00         0
         ANM       0.80      0.50      0.62         8
         CVL       0.82      0.68      0.74        53
         DAT       0.83      1.00      0.91        10
         EVT       0.64      0.88      0.74         8
         FLD       1.00      1.00      1.00         2
         LOC       0.92      1.00      0.96        12
         NUM       0.85      0.82      0.84        40
         ORG       0.81      0.88      0.84        24
         PER       0.93      0.82      0.87        34
         PLT       0.00      0.00      0.00         1
         TIM       0.00      0.00      0.00         1
         TRM       0.22      0.29      0.25         7

   micro avg       0.80      0.78      0.79       200
   macro avg       0.60      0.60      0.60       200
weighted avg       0.81      0.78      0.79       200
```

### Todo

- 추가 Pretrained 진행
- 추가 Fine-Tune 진행  
- InterActive Shell notebook 개선
 
 # References
 - [The Reformer - Pushing the limits of language modeling](https://colab.research.google.com/drive/1MYxvC4RbKeDzY2lFfesN-CvPLKLk00CQ)
 - [reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)
 - [Reformer Language Model](https://github.com/nawnoes/reformer-language-model)
 - [Kobert-NER](https://github.com/monologg/KoBERT-NER)
 - [Kober-NER-CRF](https://github.com/eagle705/pytorch-bert-crf-ner)


