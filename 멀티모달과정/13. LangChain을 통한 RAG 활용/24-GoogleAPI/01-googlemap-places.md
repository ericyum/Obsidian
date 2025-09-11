```

```python
from dotenv import load_dotenv

load_dotenv()
```


```python
import os

os.getenv("GCP_API_KEY")
```


```python
import os
import json
import googlemaps

# 환경 변수에서 GCP API 키를 가져옵니다.
# 키가 설정되지 않은 경우 스크립트를 중단합니다.
api_key = os.getenv("GCP_API_KEY")
if not api_key:
    raise ValueError(
        "GCP_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요."
    )

# API 키를 사용하여 Google Maps 클라이언트를 초기화합니다.
gmaps = googlemaps.Client(key=api_key)

try:
    # Places API의 'Text Search' 기능을 사용하여 호텔을 검색합니다.
    # 'type' 파라미터를 'lodging'으로 설정하여 숙박 시설만 검색하도록 합니다.
    print("Google Maps Places API를 통해 '서울 중구 5성급 호텔'을 검색합니다...")
    places_result = gmaps.places(
        query="서울 중구 5성급 호텔", language="ko", type="lodging"
    )

    # 검색 결과가 있는지 확인합니다.
    if places_result and places_result.get("status") == "OK":
        print(
            f"✅ 검색 성공! {len(places_result.get('results', []))}개의 장소를 찾았습니다. (상위 5개 출력)"
        )

        # 가독성을 위해 상위 5개 결과만 예쁘게 출력합니다.z
        top_5_places = places_result.get("results", [])[:5]

        # 필요한 정보만 간추려서 보여주기 위한 리스트
        simplified_results = []
        for place in top_5_places:
            simplified_results.append(
                {
                    "name": place.get("name"),
                    "address": place.get("formatted_address"),
                    "rating": place.get("rating", "N/A"),
                    "user_ratings_total": place.get("user_ratings_total", 0),
                    "place_id": place.get("place_id"),
                }
            )

        print(json.dumps(simplified_results, indent=2, ensure_ascii=False))

    else:
        print(
            f"⚠️ 호텔 정보를 찾을 수 없습니다. API 응답 상태: {places_result.get('status')}"
        )
        print("\n전체 응답:")
        print(json.dumps(places_result, indent=2, ensure_ascii=False))

except googlemaps.exceptions.ApiError as e:
    print(f"❌ Google Maps API 오류가 발생했습니다: {e}")
except Exception as e:
    print(f"스크립트 실행 중 예외가 발생했습니다: {e}")
```


```python
!pip install googlemaps
```

