# KorNeuroNER

차례
----

1. 설치
    1. 사용 환경
    2. 요구 사항
2. 전체 구조
    1. 주요 파일 구조 설명
3. 사용 방법
    1. 스크립트 실행
    2. 동작 모드
    3. 데이터 셋 구성
4. 성능 평가    

----

## 1. 설치
### 사용 환경
* 지원 OS: Linux, Windows
* 권장 메모리: 4GB 이상
* 사용 언어: Python

### 요구 사항
* Python 3.5
* Tensorflow 1.0+
* 의존 패키지
    - `JPype1`
        + 설명: JAVA 클래스 라이브러리와 Python 바인딩
        + url: https://github.com/jpype-project/jpype
    - `tensorflow` 또는 `tensorflow-gpu`
        + 설명: Python 딥러닝 라이브러리
        + url: https://www.tensorflow.org
    - `scikit-learn`
        + 설명: Python 기계학습 라이브러리
        + url: http://scikit-learn.org
    - `matplotlib`
        + 설명: Python 2차원 plotting 라이브러리
        + url: https://matplotlib.org/
    - `tqdm`
        + 설명: 반복 작업의 progress bar를 보여주는 라이브러리
        + url: https://pypi.org/project/tqdm/     

## 2. 전체 구조

### 주요 파일 구조 설명

* `src/`: 학습 모델 관련 스크립트, 모델 평가 스크립트, 입력 파이프라인 스크립트 등
    - `main.py`: 메인 실행 프로그램
    - `kor_neuroner.py`: 개체명 인식 학습 및 테스트를 수행하는 메인 모듈
    - `conlleval.py`: CoNLL 평가 방식의 결과와 리포트를 출력하는 모듈
    - `data_queue.py`: 데이터 큐 모듈
* `env/`: Python 의존 패키지

```
NeuroNER/
├── data/
│   └── kor_ner/
│       ├── train/
│       └── valid/
├── env/
│   └── requirements.txt
├── ini/
│   └── ner/
│       └── parameters.ini
├── src/
│   ├── attention.py
│   ├── bnlstm.py
│   ├── conlleval.py
│   ├── data_queue.py
│   ├── entity_lstm.py
│   ├── evaluate.py
│   ├── main.py
│   ├── metadata.py
│   ├── kor_neuroner.py
│   ├── parameters.ini
│   ├── params.py
│   ├── preprocess.py
│   ├── tagger.py
│   ├── utils.py
│   ├── utils_nlp.py
│   └── utils_plot.py
└── user_manual.md


```

## 3. 사용 방법

### 스크립트 실행
`main.py`에 학습 파라미터를 설정한 후, 실행하여 학습을 진행한다.

```
$ python3.5 main.py
```

* tensorflow 패키지 설치 시 CPU 사용, tensorflow-gpu 패키지 설치 시 GPU 사용
* `main.py` 실행 시에 설정하고 싶은 파라미터 값을 지정하거나, [`src/parameters.ini`](src/parameters.ini) 파일을 수정한다.
    - 별도의 파라미터 미지정시, [`src/parameters.ini`](src/parameters.ini) 파일의 설정된 값으로 실행
    - 커맨드 라인과 파일에 모두 설정되어 있는 경우, 커맨드 라인의 값이 우선시된다.
* `dataset_text_folder` 옵션으로 입력 데이터 셋 경로 설정
* `output_folder` 옵션으로 결과 모델 및 리포트 파일 출력 경로 설정. 입력 데이터 셋 폴더 이름으로 폴더 생성되어 출력 결과가 저장됨

* 실행 예제 (`patience` 옵션과 `token_pretrained_embedding_filepath` 옵션)
```
$ python3.5 main.py --patience=10 --token_pretrained_embedding_filepath=""
```


### 동작 모드 (operation mode)
다음과 같은 동작 모드를 제공한다.

* 학습 모드 (train from scratch)
    - 데이터 셋으로부터 학습하는 초기 학습 모드
* 학습 모드 (train from pretrained model)
    - 기존에 학습하던 모델로부터 학습을 진행하는 모드. 개체명에 레이블이 태깅되어 있지 않은 일반 텍스트 데이터 셋 학습
* 테스트 모드 (test)
    - 테스트 셋을 이용하여 학습된 모델을 테스트 하는 모드
* 인식 모드 (predict)
    - 테스트 입력 문장을 넣어서 개체명을 인식하는 모드
* vocab_expansion 모드 (vocab_expansion)
    - 학습시 보지 못 하였던 단어에 대해 인식시 처리하는 테크닉
    - 대용량 코퍼스로 미리 학습되어 사용되는 워드 임베딩과 개체명 인식기의 인코더에서 모두 학습한 단어로부터 선형 회귀 모델을 만드는 매트릭스 W를 학습한다.
    - 학습 과정에서 인코더가 보지 못 하였던 단어에 대하여, W를 사용하여 만든 임베딩을 만든다.
    - pretrained_model_folder에 확장 임베딩이 생성된다.
    - 파라미터의 `use_vocab_expansion`을 True로 하여 학습에 임베딩을 사용할 수 있다.

### 데이터 셋 구성

데이터 셋은 `dataset_text_folder`에 설정한 경로에 추가해야 한다. 데이터 셋은 CoNLL-2003 또는 BRAT 포맷의 txt 파일로 구성된다. 사용하려는 데이터 셋을 분리한 후, 각각의 셋에 맞게 폴더를 구성하여 데이터를 준비한다.

* 학습 셋: `train` 폴더에 데이터 배치
* 검증 셋: `valid` 폴더에 데이터 배치
* 테스트 셋: `test` 폴더에 데이터 배치

* 데이터 셋 구성 예제
    - `exobrain3` 데이터셋을 위한 폴더 아래에 train / valid 폴더를 구성하여 각각의 데이터 셋 배치 
```
├── data/
│   └── kor_ner/
│       ├── train/
│           └── train_korean.txt
│       └── valid/
└──         └── valid_korean.txt
```

제공되는 예제 데이터 셋은 다음과 같다

* `data/kor_ner`: 
    - 한국어 개체명 인식 말뭉치 샘플 데이터
    - 예제 문장 10개 씩, 5개의 일반 도메인 개체명 태그셋 (PS, OG, LC, DT,TI)
    - CoNLL-2003 포맷의 데이터 셋. 
    - `train_korean.txt`와 `valid_korean.txt` 파일이 있다.

* CoNLL-2003 데이터 포맷
    - Token, misc, begin, end, IOB tag 다섯 가지 정보
    - 한국어는 형태소 토큰의 종류, 공백 여부 두 가지 정보 추가
    - https://www.clips.uantwerpen.be/conll2003/ner/

데이터 포맷 구성
실제 데이터 파일은 column마다 공백 하나로 구분 된다.
```
+------+-------+-------+-----+-----------+-------+---------+
Token |  misc  | begin | end | kor_morph | space | IOB tag |
제주항공	 test	  177	 181	   1	     0	     B-OG
이	 test	  181	 182	   7	     1	      O
허위	 test	  183	 185	   1	     1	      O
할인	 test	  186	 188	   1	     0	      O
광고	 test	  188	 190	   1	     0	      O
로	 test	  190	 191	   7	     1	      O
공정위	 test	  192	 195	   1	     0	     B-OG
의	 test	  195	 196	   7	     1	      O
시정	 test	  197	 199	   1	     0	      O
명령	 test	  199	 201	   1	     0	      O
을	 test	  201	 202	   7	     1	      O
받았다  	 test  	  203	 206	   2	     0	      O
.	 test	  206	 207	   20	     1	      O

```

### 주요 파라미터 설명

`src/parameters.ini` 파일에서 여러 가지 파라미터를 수정하여 학습하거나 테스트할 수 있다.
옵션(key)와 해당 값(value) 형태로 구성되어 있고, 사용의 편의상 옵션의 목적에 따라 섹션이 구분되어 있다.
아래는 섹션 별 주요 설정을 소개한다.

* `[mode]`
    - `동작 모드`를 설정하는 섹션
    - `mode`: 학습, 검증, 테스트 등의 모드를 설정 (train, valid, test, vocab_expansion)
    - `use_vocab_expansion`: expanded vocab 임베딩 모델 사용 여부
    - `load_pretrained_model`: pretrained_model 사용 여부
* `[dataset]`
    - 입출력 데이터 경로 설정 섹션
    - `dataset_text_folder`: 입력에 사용하는 데이터 셋 폴더 경로
    - `output_folder`: 학습 및 테스트 결과가 출력되는 폴더 경로
* `[ann]`
    - 학습 네트워크 설정 섹션
    - 일반적으로 좋은 성능을 보이는 값으로 설정되면, 변경이 빈번하게 일어나지 않는 부분
    - `token_pretrained_embedding_filepath`: 학습에 사용하는 token 단위 기준 미리 학습된 임베딩 파일 경로. empty string으로 설정되면, 랜덤 초기화
    - `lstm_cell_type`: lstm에서 사용하는 memory cell의 종류 (lstm, bnlstm, lnlstm)
* `[training]`
    - 학습 진행 과정에 대한 설정 섹션
    - `patience`: early_stop이 되는 epoch 개수
    - `optimizer`: 학습에서 사용하는 최적화 함수 (sgd, adam, adadelta)
    - `dropout_rate`: dropout 확률값
* `[advanced]`
    - 여러 가지 상세 설정을 위한 섹션


## 4. 성능 평가

학습 및 테스트를 수행한 후, 성능 평가 리포트가 생성된다.
`epoch` 단위로 성능 평가가 이루어지고, 그에 대한 결과 리포트가 출력된다.
`output_folder`로 출력되는 성능 평가 결과 파일들은 다음과 같다.
평가 방식은 `conll`이 사용된다.

* `classification_report.pdf`
    -  각 개체명 태그 별 'precision', 'recall', 'f1-score' 값의 결과 리포트 PDF 파일
* `confusion_matrix.pdf`
    -  개체명 태그 분류에 대한 혼동 테이블 (confusion matrix) 리포트 PDF 파일
* `conll_evaluation.txt`
    - CoNLL-2003 포맷 데이터 셋에 대한 성능 평가 결과 txt 파일


### CoNLL-2003 평가 
CoNLL-2003 평가 지표는 개체명 태그 종류 전체에 대해서 `accuracy`, `precision`, `recall`, `FB1(f1-score)`이다. 각각의 개체명 태그 종류 별로도 평가지표를 확인할 수 있다.
현재 한국어 개체명 인식에서는 형태소 분석 단위의 토큰을 기준으로 성능을 평가하였다.

성능 평가 리포트 예제
```
processed 11228 tokens with 1248 phrases; found: 1240 phrases; correct: 1108.
accuracy:  97.27%; precision:  89.35%; recall:  88.78%; FB1:  89.07
               DT: precision:  90.84%; recall:  90.48%; FB1:  90.66  251
               LC: precision:  90.35%; recall:  89.18%; FB1:  89.76  228
               OG: precision:  83.33%; recall:  83.57%; FB1:  83.45  360
               PS: precision:  93.44%; recall:  91.94%; FB1:  92.68  366
               TI: precision:  91.43%; recall:  94.12%; FB1:  92.75  35
```