```python
from google import genai
from google.genai import types
import pathlib
import httpx

client = genai.Client()

# Retrieve and encode the PDF byte
file_path = pathlib.Path("data/RAG_25-42.pdf")

# Upload the PDF using the File API
sample_file = client.files.upload(
    file=file_path,
)

prompt = """pdf파일에서 텍스트 내용을 추출해줘. pdf에는 다양한 텍스트와 이미지, 차트가 포함되어 있어.
            텍스트만 추출해줘.
        """

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=[sample_file, prompt]
)
print(response.text)
```

### 차트를 이미지로 가져오기 위한 과정

```python
!pip install supervision
```

### 한글 프롬프트로 테스트

```python
import fitz  # PyMuPDF
import os


def pdf_to_png(pdf_path, output_path, dpi=150):
    """
    PDF 파일의 모든 페이지를 '파일명_[페이지번호].png' 형식의
    PNG 이미지들로 변환하는 함수

    Args:
        pdf_path (str): 변환할 PDF 파일 경로
        output_path (str): 출력할 PNG 파일의 경로 및 이름 형식.
                           예: 'images/result.png'
        dpi (int): 이미지 해상도 (DPI)
    """
    try:
        # 출력 경로에서 디렉토리와 기본 파일명, 확장자를 분리합니다.
        output_dir = os.path.dirname(output_path)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]

        # 출력 디렉토리가 존재하지 않으면 생성합니다.
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # PDF 문서를 엽니다.
        pdf_document = fitz.open(pdf_path)

        # PDF의 모든 페이지를 순회합니다.
        for page_num in range(len(pdf_document)):
            # 현재 페이지를 가져옵니다.
            page = pdf_document.load_page(page_num)

            # 해상도를 설정합니다.
            mat = fitz.Matrix(dpi / 72, dpi / 72)

            # 페이지를 이미지로 변환합니다.
            pix = page.get_pixmap(matrix=mat)

            # 새 출력 파일 경로를 생성합니다 (예: 'images/result_1.png').
            new_output_path = os.path.join(
                output_dir, f"{base_filename}_{page_num + 1}.png"
            )

            # PNG 파일로 저장합니다.
            pix.save(new_output_path)
            print(f"페이지 {page_num + 1} 변환 완료: {new_output_path}")

        # 메모리를 정리합니다.
        pdf_document.close()

        print(f"\nPDF가 성공적으로 PNG로 변환되었습니다.")
        return True

    except Exception as e:
        print(f"변환 오류: {e}")
        return False
```

```python
# --- 사용 예시 ---
# 멀티페이지 PDF 파일 경로
multi_page_pdf = "./data/RAG_25-42.pdf"

# 이미지가 저장될 경로와 파일명의 기본 형식을 지정합니다.
# 'output/report.png'로 지정하면 'output' 폴더에 'report_1.png', 'report_2.png'... 로 저장됩니다.
output_file_format = "output2/RAG_25-42.png"

# 함수를 호출합니다.
pdf_to_png(multi_page_pdf, output_file_format)
```

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
```

```python
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.5

IMAGE_PATH = "./output2/RAG_25-42_6.png"
PROMPT = (
    "Give the 2d bounding box chart ."  # & chart title.
    + 'Output a JSON list of the 2D bounding box in the key "box_2d", '
    + 'the text label in the key "label". Use descriptive labels.'
)
```

```python
from PIL import Image

image = Image.open(IMAGE_PATH)
width, height = image.size
target_height = int(1024 * height / width)
resized_image = image.resize((1024, target_height), Image.Resampling.LANCZOS)
# resized_image.save("./output2/RAG_25-42_6_resized.png")

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[resized_image, PROMPT],
    config=types.GenerateContentConfig(
        temperature=TEMPERATURE,
        safety_settings=safety_settings,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
```

```python
response.text
```

```python
import supervision as sv

resolution_wh = image.size

detections = sv.Detections.from_vlm(
    vlm=sv.VLM.GOOGLE_GEMINI_2_5, result=response.text, resolution_wh=resolution_wh
)

thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    smart_position=True,
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_position=sv.Position.CENTER,
)
maks_annotator = sv.MaskAnnotator()

annotated = image
for annotator in (box_annotator, label_annotator, masks_annotator):
    annotated = annotator.annotate(scene=annotated, detections=detections)

sv.plot_image(annotated)
```

```python
print(response.text)
```
