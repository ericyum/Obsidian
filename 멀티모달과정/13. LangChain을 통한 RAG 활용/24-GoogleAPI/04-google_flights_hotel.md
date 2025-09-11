```python
from dotenv import load_dotenv

load_dotenv()
```


```python
# import os
# os.getenv("SERPAPI_API_KEY")
```


### google_flights


```python
import os
import json
from serpapi import GoogleSearch

# SerpApi 클라이언트를 초기화하기 위한 파라미터를 설정합니다.
# 검색 엔진, 출발 공항 코드, 도착 공항 코드, 날짜 등을 지정합니다.
params = {
    "engine": "google_flights",
    "departure_id": "ICN",  # 인천 (Incheon)
    "arrival_id": "JFK",  # 뉴욕 (New York John F. Kennedy)
    "outbound_date": "2025-09-25",
    "return_date": "2025-09-30",
    "currency": "KRW",
    "hl": "ko",
    "api_key": os.getenv("SERPAPI_API_KEY"),
}

# 파라미터를 사용하여 GoogleSearch 객체를 생성합니다.
search = GoogleSearch(params)

try:
    # 검색을 실행하고 결과를 딕셔너리 형태로 가져옵니다.
    results = search.get_dict()

    # 결과가 있는지 확인합니다.
    if "best_flights" in results:
        print("✅ 검색 성공! 가장 저렴한 항공편 정보를 출력합니다.")
        # 가독성을 위해 JSON 형식으로 예쁘게 출력합니다.
        print(json.dumps(results["best_flights"], indent=2, ensure_ascii=False))

        # 다른 항공편 정보도 출력할 수 있습니다.
        if "other_flights" in results and results["other_flights"]:
            print("\n✈️ 다른 항공편 정보 (첫 5개):")
            other_flights_sample = results["other_flights"][:5]
            print(json.dumps(other_flights_sample, indent=2, ensure_ascii=False))

    elif "error" in results:
        print(f"❌ 검색 중 오류가 발생했습니다: {results['error']}")
    else:
        print("⚠️ 항공편 정보를 찾을 수 없습니다. 다른 날짜나 공항으로 시도해보세요.")
        print("\n전체 응답:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"스크립트 실행 중 예외가 발생했습니다: {e}")
```


## google_hotels


```python
import os
import json
from serpapi import GoogleSearch

# SerpApi 클라이언트를 초기화하기 위한 파라미터를 설정합니다.
# 검색어, 체크인/체크아웃 날짜, 인원수 등을 지정합니다.
params = {
    "engine": "google_hotels",
    "q": "서울 5성급 호텔",
    "check_in_date": "2025-10-24",
    "check_out_date": "2025-10-26",
    "adults": "2",
    "currency": "KRW",
    "hl": "ko",
    "api_key": os.getenv("SERPAPI_API_KEY")
}

# 파라미터를 사용하여 GoogleSearch 객체를 생성합니다.
search = GoogleSearch(params)

try:
    # 검색을 실행하고 결과를 딕셔너리 형태로 가져옵니다.
    results = search.get_dict()

    # 결과에 'properties' 키가 있는지 확인합니다.
    if "properties" in results:
        print("✅ 검색 성공! 호텔 정보를 출력합니다. (상위 5개)")
        
        # 상위 5개 호텔 정보만 추출하여 출력합니다.
        top_5_hotels = results["properties"][:5]
        
        # 가독성을 위해 JSON 형식으로 예쁘게 출력합니다.
        print(json.dumps(top_5_hotels, indent=2, ensure_ascii=False))

    elif "error" in results:
        print(f"❌ 검색 중 오류가 발생했습니다: {results['error']}")
    else:
        print("⚠️ 호텔 정보를 찾을 수 없습니다. 다른 검색어나 날짜로 시도해보세요.")
        print("\n전체 응답:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"스크립트 실행 중 예외가 발생했습니다: {e}")
```