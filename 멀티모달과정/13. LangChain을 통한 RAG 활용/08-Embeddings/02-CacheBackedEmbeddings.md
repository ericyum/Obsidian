# CacheBackedEmbeddings

Embeddings는 재계산을 피하기 위해 저장되거나 일시적으로 캐시될 수 있습니다.

Embeddings를 캐싱하는 것은 `CacheBackedEmbeddings`를 사용하여 수행될 수 있습니다. 캐시 지원 embedder는 embeddings를 키-값 저장소에 캐싱하는 embedder 주변에 래퍼입니다. 텍스트는 해시되고 이 해시는 캐시에서 키로 사용됩니다.

`CacheBackedEmbeddings`를 초기화하는 주요 지원 방법은 `from_bytes_store`입니다. 이는 다음 매개변수를 받습니다:

- `underlying_embeddings`: 임베딩을 위해 사용되는 embedder.
- `document_embedding_cache`: 문서 임베딩을 캐싱하기 위한 `ByteStore` 중 하나.
- `namespace`: (선택 사항, 기본값은 `""`) 문서 캐시를 위해 사용되는 네임스페이스. 이 네임스페이스는 다른 캐시와의 충돌을 피하기 위해 사용됩니다. 예를 들어, 사용된 임베딩 모델의 이름으로 설정하세요.

**주의**: 동일한 텍스트가 다른 임베딩 모델을 사용하여 임베딩될 때 충돌을 피하기 위해 `namespace` 매개변수를 설정하는 것이 중요합니다.

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

## LocalFileStore 에서 임베딩 사용 (영구 보관)

먼저, 로컬 파일 시스템을 사용하여 임베딩을 저장하고 FAISS 벡터 스토어를 사용하여 검색하는 예제를 살펴보겠습니다.

```python
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# OpenAI 임베딩을 사용하여 기본 임베딩 설정
embedding = OpenAIEmbeddings()

# 로컬 파일 저장소 설정
store = LocalFileStore("./cache/")

# 캐시를 지원하는 임베딩 생성
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embedding,
    document_embedding_cache=store,
    namespace=embedding.model,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
)
```

```python
# store에서 키들을 순차적으로 가져옵니다.
list(store.yield_keys())
```

문서를 로드하고, 청크로 분할한 다음, 각 청크를 임베딩하고 벡터 저장소에 로드합니다.

```python
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 문서 로드
raw_documents = TextLoader("./data/appendix-keywords.txt", encoding="utf-8").load()
# 문자 단위로 텍스트 분할 설정
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 문서 분할
documents = text_splitter.split_documents(raw_documents)
```

```python
# 코드 실행 시간을 측정합니다.
%time db = FAISS.from_documents(documents, cached_embedder)  # 문서로부터 FAISS 데이터베이스 생성
```

벡터 저장소를 다시 생성하려고 하면, 임베딩을 다시 계산할 필요가 없기 때문에 훨씬 더 빠르게 처리됩니다.

```python
# 캐싱된 임베딩을 사용하여 FAISS 데이터베이스 생성
%time db2 = FAISS.from_documents(documents, cached_embedder)
```

## `InmemoryByteStore` 사용 (비영구적)

다른 `ByteStore`를 사용하기 위해서는 `CacheBackedEmbeddings`를 생성할 때 해당 `ByteStore`를 사용하면 됩니다.

아래에서는, 비영구적인 `InMemoryByteStore`를 사용하여 동일한 캐시된 임베딩 객체를 생성하는 예시를 보여줍니다.

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryByteStore

store = InMemoryByteStore()  # 메모리 내 바이트 저장소 생성

# 캐시 지원 임베딩 생성
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding, store, namespace=embedding.model
)
```

