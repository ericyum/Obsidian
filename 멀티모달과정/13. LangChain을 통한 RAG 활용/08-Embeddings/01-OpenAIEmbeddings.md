# OpenAIEmbeddings

문서 임베딩은 문서의 내용을 수치적인 벡터로 변환하는 과정입니다. 

이 과정을 통해 문서의 의미를 수치화하고, 다양한 자연어 처리 작업에 활용할 수 있습니다. 대표적인 사전 학습된 언어 모델로는 BERT와 GPT가 있으며, 이러한 모델들은 문맥적 정보를 포착하여 문서의 의미를 인코딩합니다. 

문서 임베딩은 토큰화된 문서를 모델에 입력하여 임베딩 벡터를 생성하고, 이를 평균하여 전체 문서의 벡터를 생성합니다. 이 벡터는 문서 분류, 감성 분석, 문서 간 유사도 계산 등에 활용될 수 있습니다.

[더 알아보기](https://platform.openai.com/docs/guides/embeddings/embedding-models)

## 설정

먼저 langchain-openai를 설치하고 필요한 환경 변수를 설정합니다.

```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH08-Embeddings")
```

지원되는 모델 목록

| MODEL                  | PAGES PER DOLLAR | PERFORMANCE ON MTEB EVAL | MAX INPUT |
|------------------------|------------------|---------------------------|-----------|
| text-embedding-3-small | 62,500           | 62.3%                     | 8191      |
| text-embedding-3-large | 9,615            | 64.6%                     | 8191      |
| text-embedding-ada-002 | 12,500           | 61.0%                     | 8191      |


```python
from langchain_openai import OpenAIEmbeddings

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

```python
text = "임베딩 테스트를 하기 위한 샘플 문장입니다."
```

## 쿼리 임베딩

`embeddings.embed_query(text)`는 주어진 텍스트를 임베딩 벡터로 변환하는 함수입니다.

이 함수는 텍스트를 벡터 공간에 매핑하여 의미적으로 유사한 텍스트를 찾거나 텍스트 간의 유사도를 계산하는 데 사용될 수 있습니다.

```python
# 텍스트를 임베딩하여 쿼리 결과를 생성합니다.
query_result = embeddings.embed_query(text)
```

`query_result[:5]`는 `query_result` 리스트의 처음 5개 요소를 슬라이싱(slicing)하여 선택합니다.

```python
# 쿼리 결과의 처음 5개 항목을 선택합니다.
query_result[:5]
```

## Document 임베딩

`embeddings.embed_documents()` 함수를 사용하여 텍스트 문서를 임베딩합니다.

- `[text]`를 인자로 전달하여 단일 문서를 리스트 형태로 임베딩 함수에 전달합니다.
- 함수 호출 결과로 반환된 임베딩 벡터를 `doc_result` 변수에 할당합니다.


```python
doc_result = embeddings.embed_documents(
    [text, text, text, text]
)  # 텍스트를 임베딩하여 문서 벡터를 생성합니다.
```

`doc_result[0][:5]`는 `doc_result` 리스트의 첫 번째 요소에서 처음 5개의 문자를 슬라이싱하여 선택합니다.


```python
len(doc_result)  # 문서 벡터의 길이를 확인합니다.
```

```python
# 문서 결과의 첫 번째 요소에서 처음 5개 항목을 선택합니다.
doc_result[0][:5]
```

## 차원 지정

`text-embedding-3` 모델 클래스를 사용하면 반환되는 임베딩의 크기를 지정할 수 있습니다.

예를 들어, 기본적으로 `text-embedding-3-small`는 1536 차원의 임베딩을 반환합니다.


```python
# 문서 결과의 첫 번째 요소의 길이를 반환합니다.
len(doc_result[0])
```

### 차원(dimensions) 조정

하지만 `dimensions=1024`를 전달함으로써 임베딩의 크기를 1024로 줄일 수 있습니다.

```python
# OpenAI의 "text-embedding-3-small" 모델을 사용하여 1024차원의 임베딩을 생성하는 객체를 초기화합니다.
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
```

```python
# 주어진 텍스트를 임베딩하고 첫 번째 임베딩 벡터의 길이를 반환합니다.
len(embeddings_1024.embed_documents([text])[0])
```

## 유사도 계산

```python
sentence1 = "안녕하세요? 반갑습니다."
sentence2 = "안녕하세요? 반갑습니다!"
sentence3 = "안녕하세요? 만나서 반가워요."
sentence4 = "Hi, nice to meet you."
sentence5 = "I like to eat apples."
```

```python
from sklearn.metrics.pairwise import cosine_similarity

sentences = [sentence1, sentence2, sentence3, sentence4, sentence5]
embedded_sentences = embeddings_1024.embed_documents(sentences)
```

```python
def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]
```

```python
# sentence1 = "안녕하세요? 반갑습니다."
# sentence2 = "안녕하세요? 만나서 반가워요."
# sentence3 = "Hi, nice to meet you."
# sentence4 = "I like to eat apples."

for i, sentence in enumerate(embedded_sentences):
    for j, other_sentence in enumerate(embedded_sentences):
        if i < j:
            print(
                f"[유사도 {similarity(sentence, other_sentence):.4f}] {sentences[i]} \t <=====> \t {sentences[j]}"
            )
```

```