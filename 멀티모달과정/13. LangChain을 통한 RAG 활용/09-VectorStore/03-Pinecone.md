# Pinecone

Pinecone은 고성능 벡터 데이터베이스로, AI 및 머신러닝 애플리케이션을 위한 효율적인 벡터 저장 및 검색 솔루션입니다. 

Pinecone, Chroma, Faiss와 같은 벡터 데이터베이스들을 비교해보겠습니다. 

**Pinecone의 장점**

1. 확장성: 대규모 데이터셋에 대해 뛰어난 확장성을 제공합니다.
   
2. 관리 용이성: 완전 관리형 서비스로, 인프라 관리 부담이 적습니다.
   
3. 실시간 업데이트: 데이터의 실시간 삽입, 업데이트, 삭제가 가능합니다.
   
4. 고가용성: 클라우드 기반으로 높은 가용성과 내구성을 제공합니다.
   
5. API 친화적: RESTful/Python API를 통해 쉽게 통합할 수 있습니다.

**Pinecone의 단점**

1. 비용: Chroma나 Faiss에 비해 상대적으로 비용이 높을 수 있습니다.
   
2. 커스터마이징 제한: 완전 관리형 서비스이기 때문에 세부적인 커스터마이징에 제한이 있을 수 있습니다.
   
3. 데이터 위치: 클라우드에 데이터를 저장해야 하므로, 데이터 주권 문제가 있을 수 있습니다.

Chroma나 Faiss와 비교했을 때:

- Chroma/FAISS 오픈소스이며 로컬에서 실행 가능하여 초기 비용이 낮고 데이터 제어가 용이합니다. 커스터마이징의 자유도가 높습니다. 하지만 대규모 확장성 면에서는 Pinecone에 비해 제한적일 수 있습니다.

선택은 프로젝트의 규모, 요구사항, 예산 등을 고려하여 결정해야 합니다. 대규모 프로덕션 환경에서는 Pinecone이 유리할 수 있지만, 소규모 프로젝트나 실험적인 환경에서는 Chroma나 Faiss가 더 적합할 수 있습니다.

**참고**

- [Pinecone 공식 홈페이지](https://docs.pinecone.io/integrations/langchain)
- [Pinecone 랭체인](https://python.langchain.com/v0.2/docs/integrations/vectorstores/pinecone/)


```python
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
```




    True




```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH09-VectorStores")
```

    LangSmith 추적을 시작합니다.
    [프로젝트명]
    CH09-VectorStores
    

## 업데이트 안내

아래의 기능은 커스텀 구현한 내용이므로 아래의 라이브러리를 반드시 업데이트 후 진행해야 합니다.


```python
# 업데이트 명령어
# !pip install -U langchain-teddynote
```

## 한글 처리를 위한 불용어 사전

한글 불용어 사전 가져오기 (추후 토크나이저에 사용)


```python
from langchain_teddynote.korean import stopwords

# 한글 불용어 사전 불러오기 (불용어 사전 출처: https://www.ranks.nl/stopwords/korean)
stopword = stopwords()
stopword[:20]
```




    ['아',
     '휴',
     '아이구',
     '아이쿠',
     '아이고',
     '어',
     '나',
     '우리',
     '저희',
     '따라',
     '의해',
     '을',
     '를',
     '에',
     '의',
     '가',
     '으로',
     '로',
     '에게',
     '뿐이다']



## 데이터 전처리

아래는 일반 문서의 전처리 과정입니다. `ROOT_DIR` 하위에 있는 모든 `.pdf` 파일을 읽어와 `document_lsit` 에 저장합니다.


```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

split_docs = []

# 텍스트 파일을 load -> List[Document] 형태로 변환
files = sorted(glob.glob("data/*.pdf"))

for file in files:
    loader = PyMuPDFLoader(file)
    split_docs.extend(loader.load_and_split(text_splitter))

# 문서 개수 확인
len(split_docs)
```




    130




```python
split_docs[0].page_content
```




    '2023년 12월호'



Pinecone 에 DB 저장하기 위한 문서 전처리를 수행합니다. 이 과정에서 `metadata_keys` 를 지정할 수 있습니다.

추가로 metadata 를 태깅하고 싶은 경우 사전 처리 작업에서 미리 metadata 를 추가한 뒤 진행합니다.

- `split_docs`: 문서 분할 결과를 담은 List[Document] 입니다.
- `metadata_keys`: 문서에 추가할 metadata 키를 담은 List 입니다.
- `min_length`: 문서의 최소 길이를 지정합니다. 이 길이보다 짧은 문서는 제외합니다.
- `use_basename`: 소스 경로를 기준으로 파일명을 사용할지 여부를 지정합니다. 기본값은 `False` 입니다.


```python
# metadata 를 확인합니다.
split_docs[0].metadata
```




    {'producer': 'Hancom PDF 1.3.0.542',
     'creator': 'Hwp 2018 10.0.0.13462',
     'creationdate': '2023-12-08T13:28:38+09:00',
     'source': 'data\\SPRI_AI_Brief_2023년12월호_F.pdf',
     'file_path': 'data\\SPRI_AI_Brief_2023년12월호_F.pdf',
     'total_pages': 23,
     'format': 'PDF 1.4',
     'title': '',
     'author': 'dj',
     'subject': '',
     'keywords': '',
     'moddate': '2023-12-08T13:28:38+09:00',
     'trapped': '',
     'modDate': "D:20231208132838+09'00'",
     'creationDate': "D:20231208132838+09'00'",
     'page': 0}



### 문서의 전처리

- 필요한 `metadata` 정보를 추출합니다.
- 최소 길이 이상의 데이만 필터링 합니다.
  
- 문서의 `basename` 을 사용할지 여부를 지정합니다. 기본값은 `False` 입니다.
  - 여기서 `basename` 이란 파일 경로의 가장 마지막 부분을 의미합니다. 
  - 예를 들어, `/Users/teddy/data/document.pdf` 의 경우 `document.pdf` 가 됩니다.


```python
split_docs[0].metadata
```




    {'producer': 'Hancom PDF 1.3.0.542',
     'creator': 'Hwp 2018 10.0.0.13462',
     'creationdate': '2023-12-08T13:28:38+09:00',
     'source': 'data\\SPRI_AI_Brief_2023년12월호_F.pdf',
     'file_path': 'data\\SPRI_AI_Brief_2023년12월호_F.pdf',
     'total_pages': 23,
     'format': 'PDF 1.4',
     'title': '',
     'author': 'dj',
     'subject': '',
     'keywords': '',
     'moddate': '2023-12-08T13:28:38+09:00',
     'trapped': '',
     'modDate': "D:20231208132838+09'00'",
     'creationDate': "D:20231208132838+09'00'",
     'page': 0}




```python
split_docs[0].page_content
```




    '2023년 12월호'



매개변수 설명  
  
 - split_docs: 이미 분할(split)된 문서들의 리스트입니다. 보통 LangChain의 RecursiveCharacterTextSplitter 등으로 긴 문서를 작은 단위로 나눈 결과물입니다.  
  
 - metadata_keys=["source", "page", "author"]: 전처리 과정에서 보존하고 싶은 메타데이터 키(key)들을 지정합니다. 문서에 포함된 다양한 메타데이터 중, 이 키들에 해당하는 정보(예: 원본 파일명, 페이지 번호, 작성자)만 최종 metadatas에 남게 됩니다.  
  
 - min_length=5: 텍스트 내용의 최소 길이를 지정합니다. 내용이 5자보다 짧은 문서들은 불필요한 노이즈로 간주되어 전처리 과정에서 제거됩니다. 이는 임베딩의 품질을 높이는 데 도움이 됩니다.  
  
 - use_basename=True: source 메타데이터가 파일 경로인 경우, 전체 경로 대신 파일 이름만 추출할지 여부를 결정합니다. True로 설정하면 /data/documents/book1.txt와 같은 경로를 book1.txt로 단순화합니다.  
  
반환 값  
  
 - contents: 전처리된 문서의 텍스트 내용들만 담고 있는 리스트입니다.  
  
 - metadatas: metadata_keys에 지정된 정보들만 포함된 딕셔너리들의 리스트입니다. 각 딕셔너리는 contents 리스트의 해당 인덱스에 있는 문서의 메타데이터를 담고 있습니다.  


```python
from langchain_teddynote.community.pinecone import preprocess_documents

contents, metadatas = preprocess_documents(
    split_docs=split_docs,
    metadata_keys=["source", "page", "author"],
    min_length=5,
    use_basename=True,
)
```


      0%|          | 0/130 [00:00<?, ?it/s]



```python
# use_basename=True 일 때, source 키에 파일명만 저장됩니다.(디렉토리를 제외됩니다.)
metadatas["source"][:5]
```




    ['SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf']




```python
# VectorStore 에 저장할 문서 확인
contents[:5]
```




    ['2023년 12월호',
     '2023년 12월호\nⅠ. 인공지능 산업 동향 브리프\n 1. 정책/법제 \n   ▹ 미국, 안전하고 신뢰할 수 있는 AI 개발과 사용에 관한 행정명령 발표  ························· 1\n   ▹ G7, 히로시마 AI 프로세스를 통해 AI 기업 대상 국제 행동강령에 합의··························· 2\n   ▹ 영국 AI 안전성 정상회의에 참가한 28개국, AI 위험에 공동 대응 선언··························· 3',
     '▹ 미국 법원, 예술가들이 생성 AI 기업에 제기한 저작권 소송 기각····································· 4\n   ▹ 미국 연방거래위원회, 저작권청에 소비자 보호와 경쟁 측면의 AI 의견서 제출················· 5\n   ▹ EU AI 법 3자 협상, 기반모델 규제 관련 견해차로 난항··················································· 6\n \n 2. 기업/산업',
     '2. 기업/산업 \n   ▹ 미국 프런티어 모델 포럼, 1,000만 달러 규모의 AI 안전 기금 조성································ 7\n   ▹ 코히어, 데이터 투명성 확보를 위한 데이터 출처 탐색기 공개  ······································· 8\n   ▹ 알리바바 클라우드, 최신 LLM ‘통이치엔원 2.0’ 공개 ······················································ 9',
     '▹ 삼성전자, 자체 개발 생성 AI ‘삼성 가우스’ 공개 ··························································· 10\n   ▹ 구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 ················································ 11\n   ▹ IDC, 2027년 AI 소프트웨어 매출 2,500억 달러 돌파 전망··········································· 12']




```python
# VectorStore 에 저장할 metadata 확인
metadatas.keys()
```




    dict_keys(['source', 'page', 'author'])




```python
# metadata 에서 source 를 확인합니다.
metadatas["source"][:5]
```




    ['SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf',
     'SPRI_AI_Brief_2023년12월호_F.pdf']




```python
# 문서 개수 확인, 소스 개수 확인, 페이지 개수 확인
len(contents), len(metadatas["source"]), len(metadatas["page"])
```




    (130, 130, 130)



### API 키 발급

- [링크](https://app.pinecone.io/)
- 프로필 - Account - Projects - Starter - API keys - 발급

`.env` 파일에 아래와 같이 추가합니다.

```
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
```

## 새로운 VectorStore 인덱스 생성

Pinecone 의 새로운 인덱스를 생성합니다.

pinecone 인덱스를 생성합니다.

**주의사항**
- `metric` 은 유사도 측정 방법을 지정합니다. 만약 HybridSearch 를 고려하고 있다면 `metric` 은 `dotproduct` 로 지정합니다.

  create_index 함수의 dimension 매개변수는 이후에 사용할 임베딩 모델이 생성하는 벡터의 차원 수와 정확히 일치해야 합니다.  
  
  왜냐하면, Pinecone과 같은 벡터 데이터베이스는 벡터를 저장하기 위해 고정된 차원의 공간을 미리 할당해야 하기 때문입니다. 만약 모델이 생성하는 벡터의 차원(예: 1536)과 인덱스의 차원(예: 4096)이 다르면, 벡터를 데이터베이스에 삽입할 때 오류가 발생합니다.  


```python
import os
from langchain_teddynote.community.pinecone import create_index

# Pinecone 인덱스 생성
pc_index = create_index(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name="teddynote-db-index",  # 인덱스 이름을 지정합니다.
    dimension=4096,  # Embedding 차원과 맞춥니다. (OpenAIEmbeddings: 1536, UpstageEmbeddings: 4096)
    metric="dotproduct",  # 유사도 측정 방법을 지정합니다. (dotproduct, euclidean, cosine)
)
```

    [create_index]
    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'': {'vector_count': 0}},
     'total_vector_count': 0}
    

아래는 **유료 Pod** 를 사용하는 예시입니다. **유료 Pod** 는 무료 Serverless Pod 대비 더 확장된 기능을 제공합니다.

- 참고: https://docs.pinecone.io/guides/indexes/choose-a-pod-type-and-size

유료 Pod에 가입 되어있지 않으면 403에러가 난다.


```python
import os
from langchain_teddynote.community.pinecone import create_index
from pinecone import PodSpec

# Pinecone 인덱스 생성
pc_index = create_index(
    api_key=os.environ["PINECONE_API_KEY"],
    index_name="teddynote-db-index2",  # 인덱스 이름을 지정합니다.
    dimension=4096,  # Embedding 차원과 맞춥니다. (OpenAIEmbeddings: 1536, UpstageEmbeddings: 4096)
    metric="dotproduct",  # 유사도 측정 방법을 지정합니다. (dotproduct, euclidean, cosine)
    pod_spec=PodSpec(
        environment="us-west1-gcp", pod_type="p1.x1", pods=1
    ),  # 유료 Pod 사용
)
```

## Sparse Encoder 생성

- Sparse Encoder 를 생성합니다. 
- `Kiwi Tokenizer` 와 한글 불용어(stopwords) 처리를 수행합니다.
- Sparse Encoder 를 활용하여 contents 를 학습합니다. 여기서 학습한 인코드는 VectorStore 에 문서를 저장할 때 Sparse Vector 를 생성할 때 활용합니다.


```python
from langchain_teddynote.community.pinecone import (
    create_sparse_encoder,
    fit_sparse_encoder,
)

# 한글 불용어 사전 + Kiwi 형태소 분석기를 사용합니다.
sparse_encoder = create_sparse_encoder(stopwords(), mode="kiwi")
```

Sparse Encoder 에 Corpus 를 학습합니다.

- `save_path`: Sparse Encoder 를 저장할 경로입니다. 추후에 `pickle` 형식으로 저장한 Sparse Encoder 를 불러와 Query 임베딩할 때 사용합니다. 따라서, 이를 저장할 경로를 지정합니다.


```python
# Sparse Encoder 를 사용하여 contents 를 학습
saved_path = fit_sparse_encoder(
    sparse_encoder=sparse_encoder, contents=contents, save_path="./sparse_encoder.pkl"
)
```


      0%|          | 0/130 [00:00<?, ?it/s]


    [fit_sparse_encoder]
    Saved Sparse Encoder to: ./sparse_encoder.pkl
    

[선택사항] 아래는 나중에 학습하고 저장한 Sparse Encoder 를 다시 불러와야 할 때 사용하는 코드입니다.    


```python
from langchain_teddynote.community.pinecone import load_sparse_encoder

# 추후에 학습된 sparse encoder 를 불러올 때 사용합니다.
sparse_encoder = load_sparse_encoder("./sparse_encoder.pkl")
```

    [load_sparse_encoder]
    Loaded Sparse Encoder from: ./sparse_encoder.pkl
    

### Pinecone: DB Index에 추가 (Upsert)

- `context`: 문서의 내용입니다.
- `page`: 문서의 페이지 번호입니다.
- `source`: 문서의 출처입니다.
- `values`: Embedder 를 통해 얻은 문서의 임베딩입니다.
- `sparse values`: Sparse Encoder 를 통해 얻은 문서의 임베딩입니다.


```python
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings

# 임베딩 모델 생성
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
```

분산 처리를 하지 않고 배치 단위로 문서를 Upsert 합니다. 문서의 양이 많지 않다면 아래의 방식을 사용하세요.


```python
%%time
from langchain_teddynote.community.pinecone import upsert_documents
from langchain_upstage import UpstageEmbeddings

upsert_documents(
    index=pc_index,  # Pinecone 인덱스 # pc_index는 Pinecone 웹사이트에서 미리 생성해 둔 가장 큰 컨테이
    namespace="teddynote-namespace-01",  # Pinecone namespace
    contents=contents,  # 이전에 전처리한 문서 내용
    metadatas=metadatas,  # 이전에 전처리한 문서 메타데이터
    sparse_encoder=sparse_encoder,  # Sparse encoder
    embedder=upstage_embeddings,
    batch_size=32,
)
```


      0%|          | 0/5 [00:00<?, ?it/s]


    [upsert_documents]
    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-01': {'vector_count': 0}},
     'total_vector_count': 0}
    CPU times: total: 2.92 s
    Wall time: 18.2 s
    

아래는 분산처리를 수행하여 대용량 문서를 빠르게 Upsert 합니다. 대용량 업로드시 활용하세요.

 - upsert_documents: 문서를 순차적으로(하나씩) Pinecone에 업로드합니다.

 - upsert_documents_parallel: 여러 개의 작업을 동시에 실행하여 문서를 병렬로 업로드합니다.

 - max_workers=30: 병렬 처리에 사용할 작업자(Worker) 스레드의 최대 개수를 지정합니다. 30개의 작업자가 동시에 데이터를 처리하기 때문에, 순차적으로 처리하는 것보다 훨씬 빠른 속도로 업로드가 완료됩니다.


```python
%%time
from langchain_teddynote.community.pinecone import upsert_documents_parallel

upsert_documents_parallel(
    index=pc_index,  # Pinecone 인덱스
    namespace="teddynote-namespace-02",  # Pinecone namespace
    contents=contents,  # 이전에 전처리한 문서 내용
    metadatas=metadatas,  # 이전에 전처리한 문서 메타데이터
    sparse_encoder=sparse_encoder,  # Sparse encoder
    embedder=upstage_embeddings,
    batch_size=64,
    max_workers=30,
)
```


    문서 Upsert 중:   0%|          | 0/3 [00:00<?, ?it/s]


    총 130개의 Vector 가 Upsert 되었습니다.
    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-01': {'vector_count': 130},
                    'teddynote-namespace-02': {'vector_count': 66}},
     'total_vector_count': 196}
    CPU times: total: 2.08 s
    Wall time: 8.21 s
    

## 인덱스 조회/삭제

`describe_index_stats` 메서드는 인덱스의 내용에 대한 통계 정보를 제공합니다. 이 메서드를 통해 네임스페이스별 벡터 수와 차원 수 등의 정보를 얻을 수 있습니다.

**매개변수**
* `filter` (Optional[Dict[str, Union[str, float, int, bool, List, dict]]]): 특정 조건에 맞는 벡터들에 대한 통계만 반환하도록 하는 필터. 기본값은 None
* `**kwargs`: 추가 키워드 인자

**반환값**
* `DescribeIndexStatsResponse`: 인덱스에 대한 통계 정보를 담고 있는 객체

**사용 예시**
* 기본 사용: `index.describe_index_stats()`
* 필터 적용: `index.describe_index_stats(filter={'key': 'value'})`

**참고**
- metadata 필터링은 유료 사용자에 한하여 가능합니다.


```python
# 인덱스 조회
pc_index.describe_index_stats()
```




    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-01': {'vector_count': 0},
                    'teddynote-namespace-02': {'vector_count': 130}},
     'total_vector_count': 130}



### 네임스페이스(namespace) 삭제


```python
from langchain_teddynote.community.pinecone import delete_namespace

delete_namespace(
    pinecone_index=pc_index,
    namespace="teddynote-namespace-01",
)
```

    네임스페이스 'teddynote-namespace-01'의 모든 데이터가 삭제되었습니다.
    


```python
pc_index.describe_index_stats()
```




    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-02': {'vector_count': 130}},
     'total_vector_count': 130}



아래는 유료 사용자 전용 기능입니다. 유료 사용자는 metadata 필터링을 사용할 수 있습니다.


```python
from langchain_teddynote.community.pinecone import delete_by_filter

# metadata 필터링(유료 기능) 으로 삭제
delete_by_filter(
    pinecone_index=pc_index,
    namespace="teddynote-namespace-02",
    filter={"source": {"$eq": "SPRi AI Brief_8월호_산업동향.pdf"}},
)
pc_index.describe_index_stats()
```




    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-02': {'vector_count': 130}},
     'total_vector_count': 130}



## 검색기(Retriever) 생성

### PineconeKiwiHybridRetriever 초기화 파라미터 설정

`init_pinecone_index` 함수와 `PineconeKiwiHybridRetriever` 클래스는 Pinecone을 사용한 하이브리드 검색 시스템을 구현합니다. 이 시스템은 밀집 벡터와 희소 벡터를 결합하여 효과적인 문서 검색을 수행합니다.

**Pinecone 인덱스 초기화**

`init_pinecone_index` 함수는 Pinecone 인덱스를 초기화하고 필요한 구성 요소를 설정합니다.

**매개변수**
* `index_name` (str): Pinecone 인덱스 이름
* `namespace` (str): 사용할 네임스페이스
* `api_key` (str): Pinecone API 키
* `sparse_encoder_pkl_path` (str): 희소 인코더 피클 파일 경로
* `stopwords` (List[str]): 불용어 리스트
* `tokenizer` (str): 사용할 토크나이저 (기본값: "kiwi")
* `embeddings` (Embeddings): 임베딩 모델
* `top_k` (int): 반환할 최대 문서 수 (기본값: 10)
* `alpha` (float): 밀집 벡터와 희소 벡터의 가중치 조절 파라미터 (기본값: 0.5)

**주요 기능**
1. Pinecone 인덱스 초기화 및 통계 정보 출력
2. 희소 인코더(BM25) 로딩 및 토크나이저 설정
3. 네임스페이스 지정


```python
# !pip install -U pinecone langchain-teddynote
```


```python
import os
from langchain_teddynote.korean import stopwords
from langchain_teddynote.community.pinecone import init_pinecone_index
from langchain_upstage import UpstageEmbeddings

pinecone_params = init_pinecone_index(
    index_name="teddynote-db-index",  # Pinecone 인덱스 이름
    namespace="teddynote-namespace-02",  # Pinecone Namespace
    api_key=os.environ["PINECONE_API_KEY"],  # Pinecone API Key
    sparse_encoder_path="./sparse_encoder.pkl",  # Sparse Encoder 저장경로(save_path)
    stopwords=stopwords(),  # 불용어 사전
    tokenizer="kiwi",
    embeddings=UpstageEmbeddings(
        model="solar-embedding-1-large-query"
    ),  # Dense Embedder
    top_k=5,  # Top-K 문서 반환 개수
    alpha=0.5,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
)
```

    [init_pinecone_index]
    {'dimension': 4096,
     'index_fullness': 0.0,
     'namespaces': {'teddynote-namespace-02': {'vector_count': 130}},
     'total_vector_count': 130}
    

### PineconeKiwiHybridRetriever

`PineconeKiwiHybridRetriever` 클래스는 Pinecone과 Kiwi를 결합한 하이브리드 검색기를 구현합니다.

**주요 속성**
* `embeddings`: 밀집 벡터 변환용 임베딩 모델
* `sparse_encoder`: 희소 벡터 변환용 인코더
* `index`: Pinecone 인덱스 객체
* `top_k`: 반환할 최대 문서 수
* `alpha`: 밀집 벡터와 희소 벡터의 가중치 조절 파라미터
* `namespace`: Pinecone 인덱스 내 네임스페이스

**특징**
* 밀집 벡터와 희소 벡터를 결합한 HybridSearch Retriever
* 가중치 조절을 통한 검색 전략 최적화 가능
* 다양한 동적 metadata 필터링 적용 가능(`search_kwargs` 사용: `filter`, `k`, `rerank`, `rerank_model`, `top_n` 등)

**사용 예시**
1. `init_pinecone_index` 함수로 필요한 구성 요소 초기화
2. 초기화된 구성 요소로 `PineconeKiwiHybridRetriever` 인스턴스 생성
3. 생성된 검색기를 사용하여 하이브리드 검색 수행

`PineconeKiwiHybridRetriever` 를 생성합니다.


```python
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever

# 검색기 생성
pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)
```

일반 검색


```python
# 실행 결과
search_results = pinecone_retriever.invoke("gpt-4o 미니 출시 관련 정보에 대해서 알려줘")
# search_results = pinecone_retriever.invoke("삼성전자에서 자체 개발한 인공지능 모델의 이름은?")
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

    생성에서 가장 우수한 성능을 발휘
    KEY Contents
    £ 주요 LLM 중 GPT-4가 가장 환각 현상 적고 GPT-3.5 터보도 비슷한 성능 기록
    n 머신러닝 데이터 관리 기업 갈릴레오(Galileo)가 2023년 11월 15일 주요 LLM의 환각 현상을 평가한 
    ‘LLM 환각 지수(LLM Hallucination Index)’를 발표
    ∙생성 AI의 환각 현상은 AI 시스템이 잘못된 정보를 생성하거나, 현실과 다른 부정확한 결과를 내놓는
    {'author': 'dj', 'context': '생성에서 가장 우수한 성능을 발휘\nKEY Contents\n£ 주요 LLM 중 GPT-4가 가장 환각 현상 적고 GPT-3.5 터보도 비슷한 성능 기록\nn 머신러닝 데이터 관리 기업 갈릴레오(Galileo)가 2023년 11월 15일 주요 LLM의 환각 현상을 평가한 \n‘LLM 환각 지수(LLM Hallucination Index)’를 발표\n∙생성 AI의 환각 현상은 AI 시스템이 잘못된 정보를 생성하거나, 현실과 다른 부정확한 결과를 내놓는', 'page': 19.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    ▹ 구글 딥마인드, 범용 AI 모델의 기능과 동작에 대한 분류 체계 발표······························ 16
       ▹ 갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수 ··········································· 17
       
     4. 인력/교육     
       ▹ 영국 옥스퍼드 인터넷 연구소, AI 기술자의 임금이 평균 21% 높아······························· 18
       
       
     
    Ⅱ. 주요 행사
    {'author': 'dj', 'context': '▹ 구글 딥마인드, 범용 AI 모델의 기능과 동작에 대한 분류 체계 발표······························ 16\n   ▹ 갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수 ··········································· 17\n   \n 4. 인력/교육     \n   ▹ 영국 옥스퍼드 인터넷 연구소, AI 기술자의 임금이 평균 21% 높아······························· 18\n   \n   \n \nⅡ. 주요 행사', 'page': 1.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    1. 정책/법제  
    2. 기업/산업 
    3. 기술/연구 
     4. 인력/교육
    갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수
    n 주요 LLM의 환각 현상을 평가한 ‘LLM 환각 지수’에 따르면 GPT-4는 작업 유형과 관계없이 
    가장 우수한 성능을 보였으며 GPT-3.5도 거의 동등한 성능을 발휘
    n 오픈소스 모델 중에서는 메타의 라마2가 RAG 없는 질문과 답변 및 긴 형식의 텍스트 
    생성에서 가장 우수한 성능을 발휘
    KEY Contents
    {'author': 'dj', 'context': '1. 정책/법제  \n2. 기업/산업 \n3. 기술/연구 \n 4. 인력/교육\n갈릴레오의 LLM 환각 지수 평가에서 GPT-4가 가장 우수\nn 주요 LLM의 환각 현상을 평가한 ‘LLM 환각 지수’에 따르면 GPT-4는 작업 유형과 관계없이 \n가장 우수한 성능을 보였으며 GPT-3.5도 거의 동등한 성능을 발휘\nn 오픈소스 모델 중에서는 메타의 라마2가 RAG 없는 질문과 답변 및 긴 형식의 텍스트 \n생성에서 가장 우수한 성능을 발휘\nKEY Contents', 'page': 19.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    기준으로 LLM의 순위를 평가
    * 기존에 학습된 데이터가 아닌 외부 소스(데이터셋, 데이터베이스, 문서 등)에서 가져온 정보를 검색해 활용하는 기술
    n 3개의 작업 유형 평가 전체에서 오픈AI의 GPT-4가 최고의 성능을 기록했으며, GPT-3.5 터보도 
    GPT-4와 거의 동등한 성능을 발휘
    ∙메타의 라마2(Llama-2-70b)는 RAG 없는 질문과 답변 유형에서 오픈소스 모델 가운데 가장 우수했고 긴 
    형식의 텍스트 생성에서도 GPT-4에 준하는 성능을 기록했으나, RAG 포함 질문과 답변에서는 허깅
    {'author': 'dj', 'context': '기준으로 LLM의 순위를 평가\n* 기존에 학습된 데이터가 아닌 외부 소스(데이터셋, 데이터베이스, 문서 등)에서 가져온 정보를 검색해 활용하는 기술\nn 3개의 작업 유형 평가 전체에서 오픈AI의 GPT-4가 최고의 성능을 기록했으며, GPT-3.5 터보도 \nGPT-4와 거의 동등한 성능을 발휘\n∙메타의 라마2(Llama-2-70b)는 RAG 없는 질문과 답변 유형에서 오픈소스 모델 가운데 가장 우수했고 긴 \n형식의 텍스트 생성에서도 GPT-4에 준하는 성능을 기록했으나, RAG 포함 질문과 답변에서는 허깅', 'page': 19.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    기업 허깅 페이스(Hugging Face)에도 투자
    ∙구글은 챗GPT의 기반 기술과 직접 경쟁할 수 있는 차세대 LLM ‘제미니(Gemini)’를 포함한 자체 AI 
    시스템 개발에도 수십억 달러를 투자했으며, 2024년 제미니를 출시할 계획
    ☞ 출처 : The Wall Street Journal, Google Commits $2 Billion in Funding to AI Startup Anthropic, 2023.10.27.
    {'author': 'dj', 'context': '기업 허깅 페이스(Hugging Face)에도 투자\n∙구글은 챗GPT의 기반 기술과 직접 경쟁할 수 있는 차세대 LLM ‘제미니(Gemini)’를 포함한 자체 AI \n시스템 개발에도 수십억 달러를 투자했으며, 2024년 제미니를 출시할 계획\n☞ 출처 : The Wall Street Journal, Google Commits $2 Billion in Funding to AI Startup Anthropic, 2023.10.27.', 'page': 13.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    

동적 `search_kwargs` 사용
- `k`: 반환할 최대 문서 수 지정


```python
# 실행 결과
search_results = pinecone_retriever.invoke(
    "gpt-4o 미니 출시 관련 정보에 대해서 알려줘", search_kwargs={"k": 1}
    # "삼성전자에서 자체 개발한 인공지능 모델의 이름은?", search_kwargs={"k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

    생성에서 가장 우수한 성능을 발휘
    KEY Contents
    £ 주요 LLM 중 GPT-4가 가장 환각 현상 적고 GPT-3.5 터보도 비슷한 성능 기록
    n 머신러닝 데이터 관리 기업 갈릴레오(Galileo)가 2023년 11월 15일 주요 LLM의 환각 현상을 평가한 
    ‘LLM 환각 지수(LLM Hallucination Index)’를 발표
    ∙생성 AI의 환각 현상은 AI 시스템이 잘못된 정보를 생성하거나, 현실과 다른 부정확한 결과를 내놓는
    {'author': 'dj', 'context': '생성에서 가장 우수한 성능을 발휘\nKEY Contents\n£ 주요 LLM 중 GPT-4가 가장 환각 현상 적고 GPT-3.5 터보도 비슷한 성능 기록\nn 머신러닝 데이터 관리 기업 갈릴레오(Galileo)가 2023년 11월 15일 주요 LLM의 환각 현상을 평가한 \n‘LLM 환각 지수(LLM Hallucination Index)’를 발표\n∙생성 AI의 환각 현상은 AI 시스템이 잘못된 정보를 생성하거나, 현실과 다른 부정확한 결과를 내놓는', 'page': 19.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    

동적 `search_kwargs` 사용
- `alpha`: 밀집 벡터와 희소 벡터의 가중치 조절 파라미터. 0과 1 사이의 값을 지정합니다. `0.5` 가 기본값이고, 1에 가까울수록 dense 벡터의 가중치가 높아집니다.


```python
# 실행 결과
search_results = pinecone_retriever.invoke(
    "앤스로픽", search_kwargs={"alpha": 1, "k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    n 구글이 2023년 10월 27일 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 
    달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 방침
    ∙구글은 2023년 2월 앤스로픽에 이미 5억 5,000만 달러를 투자한 바 있으며, 아마존도 지난 9월 
    앤스로픽에 최대 40억 달러의 투자 계획을 공개
    ∙한편, 2023년 11월 8일 블룸버그 보도에 따르면 앤스로픽은 구글의 클라우드 서비스 사용을 위해
    {'author': 'dj', 'context': '£ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공\nn 구글이 2023년 10월 27일 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 \n달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 방침\n∙구글은 2023년 2월 앤스로픽에 이미 5억 5,000만 달러를 투자한 바 있으며, 아마존도 지난 9월 \n앤스로픽에 최대 40억 달러의 투자 계획을 공개\n∙한편, 2023년 11월 8일 블룸버그 보도에 따르면 앤스로픽은 구글의 클라우드 서비스 사용을 위해', 'page': 13.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    


```python
# 실행 결과
search_results = pinecone_retriever.invoke(
    "앤스로픽", search_kwargs={"alpha": 0, "k": 1}
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

    1. 정책/법제  
    2. 기업/산업 
    3. 기술/연구 
     4. 인력/교육
    구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 
    n 구글이 앤스로픽에 최대 20억 달러 투자에 합의하고 5억 달러를 우선 투자했으며, 앤스로픽은 
    구글과 클라우드 서비스 사용 계약도 체결
    n 3대 클라우드 사업자인 구글, 마이크로소프트, 아마존은 차세대 AI 모델의 대표 기업인 
    앤스로픽 및 오픈AI와 협력을 확대하는 추세
    KEY Contents
    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    {'author': 'dj', 'context': '1. 정책/법제  \n2. 기업/산업 \n3. 기술/연구 \n 4. 인력/교육\n구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 \nn 구글이 앤스로픽에 최대 20억 달러 투자에 합의하고 5억 달러를 우선 투자했으며, 앤스로픽은 \n구글과 클라우드 서비스 사용 계약도 체결\nn 3대 클라우드 사업자인 구글, 마이크로소프트, 아마존은 차세대 AI 모델의 대표 기업인 \n앤스로픽 및 오픈AI와 협력을 확대하는 추세\nKEY Contents\n£ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공', 'page': 13.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    

**Metadata 필터링**

동적 `search_kwargs` 사용
- `filter`: metadata 필터링 적용

(예시) `page` 가 5보다 작은 문서만 검색합니다.


```python
# 실행 결과
search_results = pinecone_retriever.invoke(
    "앤스로픽의 claude 출시 관련 내용을 알려줘",
    search_kwargs={"filter": {"page": {"$lt": 5}}, "k": 2},
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

    ▹ 삼성전자, 자체 개발 생성 AI ‘삼성 가우스’ 공개 ··························································· 10
       ▹ 구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 ················································ 11
       ▹ IDC, 2027년 AI 소프트웨어 매출 2,500억 달러 돌파 전망··········································· 12
    {'author': 'dj', 'context': '▹ 삼성전자, 자체 개발 생성 AI ‘삼성 가우스’ 공개 ··························································· 10\n   ▹ 구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 ················································ 11\n   ▹ IDC, 2027년 AI 소프트웨어 매출 2,500억 달러 돌파 전망··········································· 12', 'page': 1.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    이해관계자 협의를 통해 필요에 따라 개정할 예정
    ∙첨단 AI 시스템의 개발 과정에서 AI 수명주기 전반에 걸쳐 위험을 평가 및 완화하는 조치를 채택하고, 
    첨단 AI 시스템의 출시와 배포 이후 취약점과 오용 사고, 오용 유형을 파악해 완화
    ∙첨단 AI 시스템의 성능과 한계를 공개하고 적절하거나 부적절한 사용영역을 알리는 방법으로 투명성을 
    보장하고 책임성을 강화
    ∙산업계, 정부, 시민사회, 학계를 포함해 첨단 AI 시스템을 개발하는 조직 간 정보공유와 사고 발생 시
    {'author': 'dj', 'context': '이해관계자 협의를 통해 필요에 따라 개정할 예정\n∙첨단 AI 시스템의 개발 과정에서 AI 수명주기 전반에 걸쳐 위험을 평가 및 완화하는 조치를 채택하고, \n첨단 AI 시스템의 출시와 배포 이후 취약점과 오용 사고, 오용 유형을 파악해 완화\n∙첨단 AI 시스템의 성능과 한계를 공개하고 적절하거나 부적절한 사용영역을 알리는 방법으로 투명성을 \n보장하고 책임성을 강화\n∙산업계, 정부, 시민사회, 학계를 포함해 첨단 AI 시스템을 개발하는 조직 간 정보공유와 사고 발생 시', 'page': 4.0, 'source': 'SPRI_AI_Brief_2023년12월호_F.pdf'}
    
    ====================
    
    

동적 `search_kwargs` 사용
- `filter`: metadata 필터링 적용

(예시) `source` 가 `SPRi AI Brief_8월호_산업동향.pdf` 문서내 검색합니다.


```python
# 실행 결과
search_results = pinecone_retriever.invoke(
    "앤스로픽의 claude 3.5 출시 관련 내용을 알려줘",
    search_kwargs={
        "filter": {"source": {"$eq": "SPRi AI Brief_8월호_산업동향.pdf"}},
        "k": 3,
    },
)
for result in search_results:
    print(result.page_content)
    print(result.metadata)
    print("\n====================\n")
```

## Reranking 적용

- 동적 reranking 기능을 구현해 놓았지만, pinecone 라이브러리 의존성에 문제가 있을 수 있습니다.
- 따라서, 아래 코드는 향후 의존성 해결 후 원활하게 동작할 수 있습니다.

참고 문서: https://docs.pinecone.io/guides/inference/rerank


```python
# reranker 미사용
retrieval_results = pinecone_retriever.invoke(
    "앤스로픽의 클로드 소넷",
)

# BGE-reranker-v2-m3 모델 사용
reranked_results = pinecone_retriever.invoke(
    "앤스로픽의 클로드 소넷",
    search_kwargs={"rerank": True, "rerank_model": "bge-reranker-v2-m3", "top_n": 3},
)
```


```python
# retrieval_results 와 reranked_results 를 비교합니다.
for res1, res2 in zip(retrieval_results, reranked_results):
    print("[Retrieval]")
    print(res1.page_content)
    print("\n------------------\n")
    print("[Reranked] rerank_score: ", res2.metadata["rerank_score"])
    print(res2.page_content)
    print("\n====================\n")
```

    [Retrieval]
    4년간 30억 달러 규모의 계약을 체결
    ∙오픈AI 창업자 그룹의 일원이었던 다리오(Dario Amodei)와 다니엘라 아모데이(Daniela Amodei) 
    남매가 2021년 설립한 앤스로픽은 챗GPT의 대항마 ‘클로드(Claude)’ LLM을 개발
    n 아마존과 구글의 앤스로픽 투자에 앞서, 마이크로소프트는 차세대 AI 모델의 대표 주자인 오픈
    AI와 협력을 확대
    ∙마이크로소프트는 오픈AI에 앞서 투자한 30억 달러에 더해 2023년 1월 추가로 100억 달러를
    
    ------------------
    
    [Reranked] rerank_score:  0.12085322
    4년간 30억 달러 규모의 계약을 체결
    ∙오픈AI 창업자 그룹의 일원이었던 다리오(Dario Amodei)와 다니엘라 아모데이(Daniela Amodei) 
    남매가 2021년 설립한 앤스로픽은 챗GPT의 대항마 ‘클로드(Claude)’ LLM을 개발
    n 아마존과 구글의 앤스로픽 투자에 앞서, 마이크로소프트는 차세대 AI 모델의 대표 주자인 오픈
    AI와 협력을 확대
    ∙마이크로소프트는 오픈AI에 앞서 투자한 30억 달러에 더해 2023년 1월 추가로 100억 달러를
    
    ====================
    
    [Retrieval]
    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    n 구글이 2023년 10월 27일 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 
    달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 방침
    ∙구글은 2023년 2월 앤스로픽에 이미 5억 5,000만 달러를 투자한 바 있으며, 아마존도 지난 9월 
    앤스로픽에 최대 40억 달러의 투자 계획을 공개
    ∙한편, 2023년 11월 8일 블룸버그 보도에 따르면 앤스로픽은 구글의 클라우드 서비스 사용을 위해
    
    ------------------
    
    [Reranked] rerank_score:  0.034749627
    1. 정책/법제  
    2. 기업/산업 
    3. 기술/연구 
     4. 인력/교육
    구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 
    n 구글이 앤스로픽에 최대 20억 달러 투자에 합의하고 5억 달러를 우선 투자했으며, 앤스로픽은 
    구글과 클라우드 서비스 사용 계약도 체결
    n 3대 클라우드 사업자인 구글, 마이크로소프트, 아마존은 차세대 AI 모델의 대표 기업인 
    앤스로픽 및 오픈AI와 협력을 확대하는 추세
    KEY Contents
    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    
    ====================
    
    [Retrieval]
    1. 정책/법제  
    2. 기업/산업 
    3. 기술/연구 
     4. 인력/교육
    구글, 앤스로픽에 20억 달러 투자로 생성 AI 협력 강화 
    n 구글이 앤스로픽에 최대 20억 달러 투자에 합의하고 5억 달러를 우선 투자했으며, 앤스로픽은 
    구글과 클라우드 서비스 사용 계약도 체결
    n 3대 클라우드 사업자인 구글, 마이크로소프트, 아마존은 차세대 AI 모델의 대표 기업인 
    앤스로픽 및 오픈AI와 협력을 확대하는 추세
    KEY Contents
    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    
    ------------------
    
    [Reranked] rerank_score:  0.011869121
    £ 구글, 앤스로픽에 최대 20억 달러 투자 합의 및 클라우드 서비스 제공
    n 구글이 2023년 10월 27일 앤스로픽에 최대 20억 달러를 투자하기로 합의했으며, 이 중 5억 
    달러를 우선 투자하고 향후 15억 달러를 추가로 투자할 방침
    ∙구글은 2023년 2월 앤스로픽에 이미 5억 5,000만 달러를 투자한 바 있으며, 아마존도 지난 9월 
    앤스로픽에 최대 40억 달러의 투자 계획을 공개
    ∙한편, 2023년 11월 8일 블룸버그 보도에 따르면 앤스로픽은 구글의 클라우드 서비스 사용을 위해
    
    ====================
    
    