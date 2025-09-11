
#### 네이버 개발자 센터에서 회원가입
https://developers.naver.com/main/


```python
from dotenv import load_dotenv

load_dotenv()
```


```python
import requests
import json
import os

# 네이버 개발자 센터에서 발급받은 Client ID와 Secret
client_id = os.getenv("NAVER_CLIENT_ID")  # "YOUR_CLIENT_ID"
client_secret = os.getenv("NAVER_CLIENT_SECRET")  #

# 검색할 키워드
query = "상일동역 중화요리"

# API 요청 URL
url = f"https://openapi.naver.com/v1/search/local.json?query={query}&display=5"

# HTTP 헤더에 인증 정보 추가
headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

# API 요청 보내기
response = requests.get(url, headers=headers)
data = response.json()

# 결과 출력
print(json.dumps(data, indent=2, ensure_ascii=False))
```
