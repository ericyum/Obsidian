# Chroma

이 노트북에서는 Chroma 벡터스토어를 시작하는 방법을 다룹니다.

Chroma는 개발자의 생산성과 행복에 초점을 맞춘 AI 네이티브 오픈 소스 벡터 데이터베이스입니다. Chroma는 Apache 2.0에 따라 라이선스가 부여됩니다. 


**참고링크**

- [Chroma LangChain 문서](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/)
- [Chroma LangChain 최신 문서](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [Chroma 공식문서](https://docs.trychroma.com/getting-started)
- [LangChain 지원 VectorStore 리스트](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)

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
logging.langsmith("CH09-VectorStores")
```

샘플 데이터셋을 로드합니다.

```python
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)

# 텍스트 파일을 load -> List[Document] 형태로 변환
loader1 = TextLoader("data/nlp-keywords.txt", encoding="utf-8")
loader2 = TextLoader("data/finance-keywords.txt", encoding="utf-8")

# 문서 분할
split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

# 문서 개수 확인
len(split_doc1), len(split_doc2)
```

## VectorStore 생성

### 벡터 저장소 생성 (from_documents)

`from_documents` 클래스 메서드는 문서 리스트로부터 벡터 저장소를 생성합니다. 

**매개변수**

- `documents` (List[Document]): 벡터 저장소에 추가할 문서 리스트
- `embedding` (Optional[Embeddings]): 임베딩 함수. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None
- `collection_name` (str): 생성할 컬렉션 이름.
- `persist_directory` (Optional[str]): 컬렉션을 저장할 디렉토리. 기본값은 None
- `client_settings` (Optional[chromadb.config.Settings]): Chroma 클라이언트 설정
- `client` (Optional[chromadb.Client]): Chroma 클라이언트 인스턴스
- `collection_metadata` (Optional[Dict]): 컬렉션 구성 정보. 기본값은 None

**참고**

- `persist_directory`가 지정되면 컬렉션이 해당 디렉토리에 저장됩니다. 지정되지 않으면 데이터는 메모리에 임시로 저장됩니다.
- 이 메서드는 내부적으로 `from_texts` 메서드를 호출하여 벡터 저장소를 생성합니다.
- 문서의 `page_content`는 텍스트로, `metadata`는 메타데이터로 사용됩니다.

**반환값**

- `Chroma`: 생성된 Chroma 벡터 저장소 인스턴스

생성시 `documents` 매개변수로 `Document` 리스트를 전달합니다. embedding 에 활용할 임베딩 모델을 지정하며, `namespace` 의 역할을 하는 `collection_name` 을 지정할 수 있습니다.

`persist_directory` 지정 하지 않을 경우 메모리 공간에 저장된다.

```python
# DB 생성
db = Chroma.from_documents(
    documents=split_doc1, embedding=OpenAIEmbeddings(), collection_name="my_db"
)
```

`persist_directory` 지정시 disk 에 파일 형태로 저장합니다.

```python
# 저장할 경로 지정
DB_PATH = "./chroma_db"

# 문서를 디스크에 저장합니다. 저장시 persist_directory에 저장할 경로를 지정합니다.
persist_db = Chroma.from_documents(
    split_doc1, OpenAIEmbeddings(), persist_directory=DB_PATH, collection_name="my_db"
)
```

아래의 코드를 실행하여 `DB_PATH` 에 저장된 데이터를 로드합니다.

```python
# 디스크에서 문서를 로드합니다.
persist_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db",
)
```

불러온 VectorStore 에서 저장된 데이터를 확인합니다.

```python
# 저장된 데이터 확인
persist_db.get()
```

만약 `collection_name` 을 다르게 지정하면 저장된 데이터가 없기 때문에 아무런 결과도 얻지 못합니다.

```python
# 디스크에서 문서를 로드합니다.
persist_db2 = Chroma(
    persist_directory=DB_PATH,
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_db2",
)

# 저장된 데이터 확인
persist_db2.get()
```

### 벡터 저장소 생성 (from_texts)

`from_texts` 클래스 메서드는 텍스트 리스트로부터 벡터 저장소를 생성합니다.

**매개변수**

- `texts` (List[str]): 컬렉션에 추가할 텍스트 리스트
- `embedding` (Optional[Embeddings]): 임베딩 함수. 기본값은 None
- `metadatas` (Optional[List[dict]]): 메타데이터 리스트. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None
- `collection_name` (str): 생성할 컬렉션 이름. 기본값은 '_LANGCHAIN_DEFAULT_COLLECTION_NAME'
- `persist_directory` (Optional[str]): 컬렉션을 저장할 디렉토리. 기본값은 None
- `client_settings` (Optional[chromadb.config.Settings]): Chroma 클라이언트 설정
- `client` (Optional[chromadb.Client]): Chroma 클라이언트 인스턴스
- `collection_metadata` (Optional[Dict]): 컬렉션 구성 정보. 기본값은 None

**참고**

- `persist_directory`가 지정되면 컬렉션이 해당 디렉토리에 저장됩니다. 지정되지 않으면 데이터는 메모리에 임시로 저장됩니다.
- `ids`가 제공되지 않으면 UUID를 사용하여 자동으로 생성됩니다.

**반환값**

- 생성된 벡터 저장소 인스턴스

```python
# 문자열 리스트로 생성
db2 = Chroma.from_texts(
    ["안녕하세요. 정말 반갑습니다.", "제 이름은 테디입니다."],
    embedding=OpenAIEmbeddings(),
)
```

```python
# 데이터를 조회합니다.
db2.get()
```

### 유사도 검색

`similarity_search` 메서드는 Chroma 데이터베이스에서 유사도 검색을 수행합니다. 이 메서드는 주어진 쿼리와 가장 유사한 문서들을 반환합니다.

**매개변수**

- `query` (str): 검색할 쿼리 텍스트
- `k` (int, 선택적): 반환할 결과의 수. 기본값은 4입니다.
- `filter` (Dict[str, str], 선택적): 메타데이터로 필터링. 기본값은 None입니다.

**참고**

- `k` 값을 조절하여 원하는 수의 결과를 얻을 수 있습니다.
- `filter` 매개변수를 사용하여 특정 메타데이터 조건에 맞는 문서만 검색할 수 있습니다.
- 이 메서드는 점수 정보 없이 문서만 반환합니다. 점수 정보도 필요한 경우 `similarity_search_with_score` 메서드를 직접 사용하세요.

**반환값**

- `List[Document]`: 쿼리 텍스트와 가장 유사한 문서들의 리스트

```python
db.similarity_search("TF IDF 에 대하여 알려줘")
```

`k` 값에 검색 결과의 개수를 지정할 수 있습니다.

```python
db.similarity_search("TF IDF 에 대하여 알려줘", k=2)
```

`filter` 에 `metadata` 정보를 활용하여 검색 결과를 필터링 할 수 있습니다.

```python
# filter 사용
db.similarity_search(
    "TF IDF 에 대하여 알려줘", filter={"source": "data/nlp-keywords.txt"}, k=2
)
```

다음은 `filter` 에서 다른 `source` 를 사용하여 검색한 결과를 확인합니다.

```python
# filter 사용
db.similarity_search(
    "TF IDF 에 대하여 알려줘", filter={"source": "data/finance-keywords.txt"}, k=2
)
```

### 벡터 저장소에 문서 추가

`add_documents` 메서드는 벡터 저장소에 문서를 추가하거나 업데이트합니다.

**매개변수**

- `documents` (List[Document]): 벡터 저장소에 추가할 문서 리스트
- `**kwargs`: 추가 키워드 인자
  - `ids`: 문서 ID 리스트 (제공 시 문서의 ID보다 우선함)

**참고**

- `add_texts` 메서드가 구현되어 있어야 합니다.
- 문서의 `page_content`는 텍스트로, `metadata`는 메타데이터로 사용됩니다.
- 문서에 ID가 있고 `kwargs`에 ID가 제공되지 않으면 문서의 ID가 사용됩니다.
- `kwargs`의 ID와 문서 수가 일치하지 않으면 ValueError가 발생합니다.

**반환값**

- `List[str]`: 추가된 텍스트의 ID 리스트

**예외**

- `NotImplementedError`: `add_texts` 메서드가 구현되지 않은 경우 발생

```python
from langchain_core.documents import Document

# page_content, metadata, id 지정
db.add_documents(
    [
        Document(
            page_content="안녕하세요! 이번엔 도큐먼트를 새로 추가해 볼께요",
            metadata={"source": "mydata.txt"},
            id="1",
        )
    ]
)
```

```python
# id=1 로 문서 조회
db.get("1")
```

`add_texts` 메서드는 텍스트를 임베딩하고 벡터 저장소에 추가합니다.

**매개변수**

- `texts` (Iterable[str]): 벡터 저장소에 추가할 텍스트 리스트
- `metadatas` (Optional[List[dict]]): 메타데이터 리스트. 기본값은 None
- `ids` (Optional[List[str]]): 문서 ID 리스트. 기본값은 None

**참고**

- `ids`가 제공되지 않으면 UUID를 사용하여 자동으로 생성됩니다.
- 임베딩 함수가 설정되어 있으면 텍스트를 임베딩합니다.
- 메타데이터가 제공된 경우:
  - 메타데이터가 있는 텍스트와 없는 텍스트를 분리하여 처리합니다.
  - 메타데이터가 없는 텍스트의 경우 빈 딕셔너리로 채웁니다.
- 컬렉션에 upsert 작업을 수행하여 텍스트, 임베딩, 메타데이터를 추가합니다.

**반환값**

- `List[str]`: 추가된 텍스트의 ID 리스트

**예외**

- `ValueError`: 복잡한 메타데이터로 인한 오류 발생 시, 필터링 방법 안내 메시지와 함께 발생

기존의 아이디에 추가하는 경우 `upsert` 가 수행되며, 기존의 문서는 대체됩니다.

```python
# 신규 데이터를 추가합니다. 이때 기존의 id=1 의 데이터는 덮어쓰게 됩니다.
db.add_texts(
    ["이전에 추가한 Document 를 덮어쓰겠습니다.", "덮어쓴 결과가 어떤가요?"],
    metadatas=[{"source": "mydata.txt"}, {"source": "mydata.txt"}],
    ids=["1", "2"],
)
```

```python
# id=1 조회
db.get(["1"])
```

### 벡터 저장소에서 문서 삭제

`delete` 메서드는 벡터 저장소에서 지정된 ID의 문서를 삭제합니다.

**매개변수**

- `ids` (Optional[List[str]]): 삭제할 문서의 ID 리스트. 기본값은 None

**참고**

- 이 메서드는 내부적으로 컬렉션의 `delete` 메서드를 호출합니다.
- `ids`가 None이면 아무 작업도 수행하지 않습니다.

**반환값**

- None

```python
# id 1 삭제
db.delete(ids=["1"])
```

```python
# 문서 조회
db.get(["1", "2"])
```

```python
# where 조건으로 metadata 조회
db.get(where={"source": "mydata.txt"})
```

### 초기화(reset_collection)

`reset_collection` 메서드는 벡터 저장소의 컬렉션을 초기화합니다.

```python
# 컬렉션 초기화
db.reset_collection()
```

```python
# 초기화 후 문서 조회
db.get()
```

### 벡터 저장소를 검색기(Retriever)로 변환

`as_retriever` 메서드는 벡터 저장소를 기반으로 VectorStoreRetriever를 생성합니다.

**매개변수**

- `**kwargs`: 검색 함수에 전달할 키워드 인자
  - `search_type` (Optional[str]): 검색 유형 (`"similarity"`, `"mmr"`, `"similarity_score_threshold"`)
  - `search_kwargs` (Optional[Dict]): 검색 함수에 전달할 추가 인자
    - `k`: 반환할 문서 수 (기본값: 4)
    - `score_threshold`: 최소 유사도 임계값
    - `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
    - `lambda_mult`: MMR 결과의 다양성 조절 (0~1, 기본값: 0.5)
    - `filter`: 문서 메타데이터 필터링

**반환값**

- `VectorStoreRetriever`: 벡터 저장소 기반 검색기 인스턴스

`DB` 를 생성합니다.

```python
# DB 생성
db = Chroma.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=OpenAIEmbeddings(),
    collection_name="nlp",
)
```

기본 값으로 설정된 4개 문서를 유사도 검색을 수행하여 조회합니다.

```python
retriever = db.as_retriever()
retriever.invoke("Word2Vec 에 대하여 알려줘")
```

다양성이 높은 더 많은 문서 검색

- `k`: 반환할 문서 수 (기본값: 4)
- `fetch_k`: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
- `lambda_mult`: MMR 결과의 다양성 조절 (0~1, 기본값: 0.5, 0: 유사도 점수만 고려, 1: 다양성만 고려)

```python
retriever = db.as_retriever(
    # search_type="similarity", search_kwargs={"k": 2}
    search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25, "fetch_k": 10}
)
retriever.invoke("Word2Vec 에 대하여 알려줘")
```

MMR 알고리즘을 위해 더 많은 문서를 가져오되 상위 2개만 반환

```python
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})
retriever.invoke("Word2Vec 에 대하여 알려줘")
```

특정 임계값 이상의 유사도를 가진 문서만 검색

```python
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8}
)

retriever.invoke("Word2Vec 에 대하여 알려줘")
```

가장 유사한 단일 문서만 검색

```python
retriever = db.as_retriever(search_kwargs={"k": 1})

retriever.invoke("Word2Vec 에 대하여 알려줘")
```

특정 메타데이터 필터 적용

```python
retriever = db.as_retriever(
    search_kwargs={"filter": {"source": "data/finance-keywords.txt"}, "k": 2}
)
retriever.invoke("ESG 에 대하여 알려줘")
```

## 멀티모달 검색

Chroma는 멀티모달 컬렉션, 즉 여러 양식의 데이터를 포함하고 쿼리할 수 있는 컬렉션을 지원합니다.

## 데이터 세트

허깅페이스에서 호스팅되는 [coco object detection dataset](https://huggingface.co/datasets/detection-datasets/coco)의 작은 하위 집합을 사용합니다.

데이터 세트의 모든 이미지 중 일부만 로컬로 다운로드하고 이를 사용하여 멀티모달 컬렉션을 생성합니다.

```python
import os
from datasets import load_dataset
from matplotlib import pyplot as plt

# COCO 데이터셋 로드
dataset = load_dataset(
    path="detection-datasets/coco", name="default", split="train", streaming=True
)

# 이미지 저장 폴더와 이미지 개수 설정
IMAGE_FOLDER = "tmp"
N_IMAGES = 20

# 그래프 플로팅을 위한 설정
plot_cols = 5
plot_rows = N_IMAGES // plot_cols
fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_rows * 2, plot_cols * 2))
axes = axes.flatten()

# 이미지를 폴더에 저장하고 그래프에 표시
dataset_iter = iter(dataset)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
for i in range(N_IMAGES):
    # 데이터셋에서 이미지와 레이블 추출
    data = next(dataset_iter)
    image = data["image"]
    label = data["objects"]["category"][0]  # 첫 번째 객체의 카테고리를 레이블로 사용

    # 그래프에 이미지 표시 및 레이블 추가
    axes[i].imshow(image)
    axes[i].set_title(label, fontsize=8)
    axes[i].axis("off")

    # 이미지 파일로 저장
    image.save(f"{IMAGE_FOLDER}/{i}.jpg")

# 그래프 레이아웃 조정 및 표시
plt.tight_layout()
plt.show()
```

![](./images/chroma-01.png)

### Multimodal Embeddings

Multimodal Embeddings 을 활용하여 이미지, 텍스트에 대한 Embedding 을 생성합니다.

이번 튜토리얼에서는 OpenClipEmbeddingFunction 을 사용하여 이미지를 임베딩합니다.

- [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main)

### Model 벤치마크

| Model                              | Training data  | Resolution | # of samples seen | ImageNet zero-shot acc. |
|------------------------------------|----------------|------------|-------------------|-------------------------|
| ConvNext-Base                       | LAION-2B       | 256px      | 13B               | 71.5%                   |
| ConvNext-Large                      | LAION-2B       | 320px      | 29B               | 76.9%                   |
| ConvNext-XXLarge                    | LAION-2B       | 256px      | 34B               | 79.5%                   |
| ViT-B/32                            | DataComp-1B    | 256px      | 34B               | 72.8%                   |
| ViT-B/16                            | DataComp-1B    | 224px      | 13B               | 73.5%                   |
| ViT-L/14                            | LAION-2B       | 224px      | 32B               | 75.3%                   |
| ViT-H/14                            | LAION-2B       | 224px      | 32B               | 78.0%                   |
| ViT-L/14                            | DataComp-1B    | 224px      | 13B               | 79.2%                   |
| ViT-G/14                            | LAION-2B       | 224px      | 34B               | 80.1%                   |
| ViT-L/14 ([Original CLIP](https://openai.com/research/clip)) | WIT            | 224px      | 13B               | 75.5%                   |
| ViT-SO400M/14 ([SigLIP](https://github.com/mlfoundations/open_clip)) | WebLI | 224px | 45B | 82.0% |
| ViT-SO400M-14-SigLIP-384 ([SigLIP](https://github.com/mlfoundations/open_clip)) | WebLI | 384px | 45B | 83.1% |
| ViT-H/14-quickgelu ([DFN](https://www.deeplearning.ai/glossary/neural-networks/)) | DFN-5B | 224px | 39B | 83.4% |
| ViT-H-14-378-quickgelu ([DFN](https://www.deeplearning.ai/glossary/neural-networks/)) | DFN-5B | 378px | 44B | 84.4% |

아래의 예시에서 `model_name` 과 `checkpoint` 를 설정하여 사용합니다.

- `model_name`: OpenCLIP 모델명
- `checkpoint`: OpenCLIP 모델의 `Training data` 에 해당하는 이름

```python
import open_clip
import pandas as pd

# 사용 가능한 모델/Checkpoint 를 출력
pd.DataFrame(open_clip.list_pretrained(), columns=["model_name", "checkpoint"]).head(10)
```

```python
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# OpenCLIP 임베딩 함수 객체 생성
image_embedding_function = OpenCLIPEmbeddings(
    model_name="ViT-H-14-378-quickgelu", checkpoint="dfn5b"
)
```

이미지의 경로를 list 로 저장합니다.

```python
# 이미지의 경로를 리스트로 저장
image_uris = sorted(
    [
        os.path.join("tmp", image_name)
        for image_name in os.listdir("tmp")
        if image_name.endswith(".jpg")
    ]
)

image_uris
```

```python
from langchain_teddynote.models import MultiModal
from langchain_openai import ChatOpenAI

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini")

# MultiModal 모델 설정
model = MultiModal(
    model=llm,
    system_prompt="Your mission is to describe the image in detail",  # 시스템 프롬프트: 이미지를 상세히 설명하도록 지시
    user_prompt="Description should be written in one sentence(less than 60 characters)",  # 사용자 프롬프트: 60자 이내의 한 문장으로 설명 요청
)
```

image 에 대한 description 을 생성합니다.

```python
# 이미지 설명 생성
model.invoke(image_uris[0])
```

![](./images/chroma-02.png)

```python
# 이미지 설명
descriptions = dict()

for image_uri in image_uris:
    descriptions[image_uri] = model.invoke(image_uri, display_image=False)

# 생성된 결과물 출력
descriptions
```

```python
import os
from PIL import Image
import matplotlib.pyplot as plt

# 원본 이미지, 처리된 이미지, 텍스트 설명을 저장할 리스트 초기화
original_images = []
images = []
texts = []

# 그래프 크기 설정 (20x10 인치)
plt.figure(figsize=(20, 10))

# 'tmp' 디렉토리에 저장된 이미지 파일들을 처리
for i, image_uri in enumerate(image_uris):
    # 이미지 파일 열기 및 RGB 모드로 변환
    image = Image.open(image_uri).convert("RGB")

    # 4x5 그리드의 서브플롯 생성
    plt.subplot(4, 5, i + 1)

    # 이미지 표시
    plt.imshow(image)

    # 이미지 파일명과 설명을 제목으로 설정
    plt.title(f"{os.path.basename(image_uri)}\n{descriptions[image_uri]}", fontsize=8)

    # x축과 y축의 눈금 제거
    plt.xticks([])
    plt.yticks([])

    # 원본 이미지, 처리된 이미지, 텍스트 설명을 각 리스트에 추가
    original_images.append(image)
    images.append(image)
    texts.append(descriptions[image_uri])

# 서브플롯 간 간격 조정
plt.tight_layout()
```

![](./images/chroma-03.png)

아래는 생성한 이미지 description 과 텍스트 간의 유사도를 계산합니다.

```python
import numpy as np

# 이미지와 텍스트 임베딩
# 이미지 URI를 사용하여 이미지 특징 추출
img_features = image_embedding_function.embed_image(image_uris)
# 텍스트 설명에 "This is" 접두사를 추가하고 텍스트 특징 추출
text_features = image_embedding_function.embed_documents(
    ["This is " + desc for desc in texts]
)

# 행렬 연산을 위해 리스트를 numpy 배열로 변환
img_features_np = np.array(img_features)
text_features_np = np.array(text_features)

# 유사도 계산
# 텍스트와 이미지 특징 간의 코사인 유사도를 계산
similarity = np.matmul(text_features_np, img_features_np.T)
```

텍스트 대 이미지 description 간 유사도를 구하고 시각화합니다.

```python
# 유사도 행렬을 시각화하기 위한 플롯 생성
count = len(descriptions)
plt.figure(figsize=(20, 14))

# 유사도 행렬을 히트맵으로 표시
plt.imshow(similarity, vmin=0.1, vmax=0.3, cmap="coolwarm")
plt.colorbar()  # 컬러바 추가

# y축에 텍스트 설명 표시
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])  # x축 눈금 제거

# 원본 이미지를 x축 아래에 표시
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

# 유사도 값을 히트맵 위에 텍스트로 표시
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

# 플롯 테두리 제거
for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

# 플롯 범위 설정
plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

# 제목 추가
plt.title("Cosine Similarity", size=20)
```

![](./images/chroma-04.png)

### Vectorstore 생성 및 이미지 추가

Vectorstore 를 생성하고 이미지를 추가합니다.

```python
# DB 생성
image_db = Chroma(
    collection_name="multimodal",
    embedding_function=image_embedding_function,
)

# 이미지 추가
image_db.add_images(uris=image_uris)
```

아래는 이미지 검색된 결과를 이미지로 출력하기 위한 helper class 입니다.

```python
import base64
import io
from PIL import Image
from IPython.display import HTML, display
from langchain.schema import Document


class ImageRetriever:
    def __init__(self, retriever):
        """
        이미지 검색기를 초기화합니다.

        인자:
        retriever: LangChain의 retriever 객체
        """
        self.retriever = retriever

    def invoke(self, query):
        """
        쿼리를 사용하여 이미지를 검색하고 표시합니다.

        인자:
        query (str): 검색 쿼리
        """
        docs = self.retriever.invoke(query)
        if docs and isinstance(docs[0], Document):
            self.plt_img_base64(docs[0].page_content)
        else:
            print("검색된 이미지가 없습니다.")
        return docs

    @staticmethod
    def resize_base64_image(base64_string, size=(224, 224)):
        """
        Base64 문자열로 인코딩된 이미지의 크기를 조정합니다.

        인자:
        base64_string (str): 원본 이미지의 Base64 문자열.
        size (tuple): (너비, 높이)로 표현된 원하는 이미지 크기.

        반환:
        str: 크기가 조정된 이미지의 Base64 문자열.
        """
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def plt_img_base64(img_base64):
        """
        Base64로 인코딩된 이미지를 표시합니다.

        인자:
        img_base64 (str): Base64로 인코딩된 이미지 문자열
        """
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        display(HTML(image_html))
```
```