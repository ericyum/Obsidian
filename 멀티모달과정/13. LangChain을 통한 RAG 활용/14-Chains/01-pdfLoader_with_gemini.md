```python
# !pip install google
```

```python
# !pip install google.genai
```

```python
from google import genai
from google.genai import types
import pathlib
import httpx

client = genai.Client()

# Retrieve and encode the PDF byte
file_path = pathlib.Path("data/deoksugung_pamphlet.pdf")


# Upload the PDF using the File API
sample_file = client.files.upload(
    file=file_path,
)

prompt = """pdf파일에서 텍스트 내용을 추출해줘. pdf는 페이지가 가로로 4페이지가 나란히 배치되어 있어.
            왼쪽페이지부터 차례로 읽어주고 불필요한 정보는 삭제해줘.
        """

response = client.models.generate_content(
    model="gemini-2.5-pro", contents=[sample_file, prompt]
)
print(response.text)
```
