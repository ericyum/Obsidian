참조 :
https://python.langchain.com/docs/integrations/text_embedding/open_clip/

# OpenClip

[OpenClip](https://github.com/mlfoundations/open_clip/tree/main) is an source implementation of OpenAI's CLIP.

These multi-modal embeddings can be used to embed images or text.


```python
#%pip install --upgrade --quiet  langchain-experimental
```


```python
#%pip install --upgrade --quiet  pillow open_clip_torch torch matplotlib
```

We can the list of available CLIP embedding models and checkpoints:


```python
import open_clip

open_clip.list_pretrained()
```




    [('RN50', 'openai'),
     ('RN50', 'yfcc15m'),
     ('RN50', 'cc12m'),
     ('RN101', 'openai'),
     ('RN101', 'yfcc15m'),
     ('RN50x4', 'openai'),
     ('RN50x16', 'openai'),
     ('RN50x64', 'openai'),
     ('ViT-B-32', 'openai'),
     ('ViT-B-32', 'laion400m_e31'),
     ('ViT-B-32', 'laion400m_e32'),
     ('ViT-B-32', 'laion2b_e16'),
     ('ViT-B-32', 'laion2b_s34b_b79k'),
     ('ViT-B-32', 'datacomp_xl_s13b_b90k'),
     ('ViT-B-32', 'datacomp_m_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_clip_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_laion_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_image_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_text_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_basic_s128m_b4k'),
     ('ViT-B-32', 'commonpool_m_s128m_b4k'),
     ('ViT-B-32', 'datacomp_s_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_clip_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_laion_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_image_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_text_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_basic_s13m_b4k'),
     ('ViT-B-32', 'commonpool_s_s13m_b4k'),
     ('ViT-B-32', 'metaclip_400m'),
     ('ViT-B-32', 'metaclip_fullcc'),
     ('ViT-B-32-256', 'datacomp_s34b_b86k'),
     ('ViT-B-16', 'openai'),
     ('ViT-B-16', 'laion400m_e31'),
     ('ViT-B-16', 'laion400m_e32'),
     ('ViT-B-16', 'laion2b_s34b_b88k'),
     ('ViT-B-16', 'datacomp_xl_s13b_b90k'),
     ('ViT-B-16', 'datacomp_l_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_clip_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_laion_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_image_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_text_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_basic_s1b_b8k'),
     ('ViT-B-16', 'commonpool_l_s1b_b8k'),
     ('ViT-B-16', 'dfn2b'),
     ('ViT-B-16', 'metaclip_400m'),
     ('ViT-B-16', 'metaclip_fullcc'),
     ('ViT-B-16-plus-240', 'laion400m_e31'),
     ('ViT-B-16-plus-240', 'laion400m_e32'),
     ('ViT-L-14', 'openai'),
     ('ViT-L-14', 'laion400m_e31'),
     ('ViT-L-14', 'laion400m_e32'),
     ('ViT-L-14', 'laion2b_s32b_b82k'),
     ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
     ('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'),
     ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'),
     ('ViT-L-14', 'commonpool_xl_s13b_b90k'),
     ('ViT-L-14', 'metaclip_400m'),
     ('ViT-L-14', 'metaclip_fullcc'),
     ('ViT-L-14', 'dfn2b'),
     ('ViT-L-14', 'dfn2b_s39b'),
     ('ViT-L-14-336', 'openai'),
     ('ViT-H-14', 'laion2b_s32b_b79k'),
     ('ViT-H-14', 'metaclip_fullcc'),
     ('ViT-H-14', 'metaclip_altogether'),
     ('ViT-H-14', 'dfn5b'),
     ('ViT-H-14-378', 'dfn5b'),
     ('ViT-g-14', 'laion2b_s12b_b42k'),
     ('ViT-g-14', 'laion2b_s34b_b88k'),
     ('ViT-bigG-14', 'laion2b_s39b_b160k'),
     ('ViT-bigG-14', 'metaclip_fullcc'),
     ('roberta-ViT-B-32', 'laion2b_s12b_b32k'),
     ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'),
     ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'),
     ('convnext_base', 'laion400m_s13b_b51k'),
     ('convnext_base_w', 'laion2b_s13b_b82k'),
     ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
     ('convnext_base_w', 'laion_aesthetic_s13b_b82k'),
     ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
     ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
     ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
     ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
     ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
     ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
     ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),
     ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'),
     ('coca_ViT-B-32', 'laion2b_s13b_b90k'),
     ('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),
     ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
     ('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'),
     ('EVA01-g-14', 'laion400m_s11b_b41k'),
     ('EVA01-g-14-plus', 'merged2b_s11b_b114k'),
     ('EVA02-B-16', 'merged2b_s8b_b131k'),
     ('EVA02-L-14', 'merged2b_s4b_b131k'),
     ('EVA02-L-14-336', 'merged2b_s6b_b61k'),
     ('EVA02-E-14', 'laion2b_s4b_b115k'),
     ('EVA02-E-14-plus', 'laion2b_s9b_b144k'),
     ('ViT-B-16-SigLIP', 'webli'),
     ('ViT-B-16-SigLIP-256', 'webli'),
     ('ViT-B-16-SigLIP-i18n-256', 'webli'),
     ('ViT-B-16-SigLIP-384', 'webli'),
     ('ViT-B-16-SigLIP-512', 'webli'),
     ('ViT-L-16-SigLIP-256', 'webli'),
     ('ViT-L-16-SigLIP-384', 'webli'),
     ('ViT-SO400M-14-SigLIP', 'webli'),
     ('ViT-SO400M-16-SigLIP-i18n-256', 'webli'),
     ('ViT-SO400M-14-SigLIP-378', 'webli'),
     ('ViT-SO400M-14-SigLIP-384', 'webli'),
     ('ViT-B-32-SigLIP2-256', 'webli'),
     ('ViT-B-16-SigLIP2', 'webli'),
     ('ViT-B-16-SigLIP2-256', 'webli'),
     ('ViT-B-16-SigLIP2-384', 'webli'),
     ('ViT-B-16-SigLIP2-512', 'webli'),
     ('ViT-L-16-SigLIP2-256', 'webli'),
     ('ViT-L-16-SigLIP2-384', 'webli'),
     ('ViT-L-16-SigLIP2-512', 'webli'),
     ('ViT-SO400M-14-SigLIP2', 'webli'),
     ('ViT-SO400M-14-SigLIP2-378', 'webli'),
     ('ViT-SO400M-16-SigLIP2-256', 'webli'),
     ('ViT-SO400M-16-SigLIP2-384', 'webli'),
     ('ViT-SO400M-16-SigLIP2-512', 'webli'),
     ('ViT-gopt-16-SigLIP2-256', 'webli'),
     ('ViT-gopt-16-SigLIP2-384', 'webli'),
     ('ViT-L-14-CLIPA', 'datacomp1b'),
     ('ViT-L-14-CLIPA-336', 'datacomp1b'),
     ('ViT-H-14-CLIPA', 'datacomp1b'),
     ('ViT-H-14-CLIPA-336', 'laion2b'),
     ('ViT-H-14-CLIPA-336', 'datacomp1b'),
     ('ViT-bigG-14-CLIPA', 'datacomp1b'),
     ('ViT-bigG-14-CLIPA-336', 'datacomp1b'),
     ('nllb-clip-base', 'v1'),
     ('nllb-clip-large', 'v1'),
     ('nllb-clip-base-siglip', 'v1'),
     ('nllb-clip-base-siglip', 'mrl'),
     ('nllb-clip-large-siglip', 'v1'),
     ('nllb-clip-large-siglip', 'mrl'),
     ('MobileCLIP-S1', 'datacompdr'),
     ('MobileCLIP-S2', 'datacompdr'),
     ('MobileCLIP-B', 'datacompdr'),
     ('MobileCLIP-B', 'datacompdr_lt'),
     ('ViTamin-S', 'datacomp1b'),
     ('ViTamin-S-LTT', 'datacomp1b'),
     ('ViTamin-B', 'datacomp1b'),
     ('ViTamin-B-LTT', 'datacomp1b'),
     ('ViTamin-L', 'datacomp1b'),
     ('ViTamin-L-256', 'datacomp1b'),
     ('ViTamin-L-336', 'datacomp1b'),
     ('ViTamin-L-384', 'datacomp1b'),
     ('ViTamin-L2', 'datacomp1b'),
     ('ViTamin-L2-256', 'datacomp1b'),
     ('ViTamin-L2-336', 'datacomp1b'),
     ('ViTamin-L2-384', 'datacomp1b'),
     ('ViTamin-XL-256', 'datacomp1b'),
     ('ViTamin-XL-336', 'datacomp1b'),
     ('ViTamin-XL-384', 'datacomp1b'),
     ('RN50-quickgelu', 'openai'),
     ('RN50-quickgelu', 'yfcc15m'),
     ('RN50-quickgelu', 'cc12m'),
     ('RN101-quickgelu', 'openai'),
     ('RN101-quickgelu', 'yfcc15m'),
     ('RN50x4-quickgelu', 'openai'),
     ('RN50x16-quickgelu', 'openai'),
     ('RN50x64-quickgelu', 'openai'),
     ('ViT-B-32-quickgelu', 'openai'),
     ('ViT-B-32-quickgelu', 'laion400m_e31'),
     ('ViT-B-32-quickgelu', 'laion400m_e32'),
     ('ViT-B-32-quickgelu', 'metaclip_400m'),
     ('ViT-B-32-quickgelu', 'metaclip_fullcc'),
     ('ViT-B-16-quickgelu', 'openai'),
     ('ViT-B-16-quickgelu', 'dfn2b'),
     ('ViT-B-16-quickgelu', 'metaclip_400m'),
     ('ViT-B-16-quickgelu', 'metaclip_fullcc'),
     ('ViT-L-14-quickgelu', 'openai'),
     ('ViT-L-14-quickgelu', 'metaclip_400m'),
     ('ViT-L-14-quickgelu', 'metaclip_fullcc'),
     ('ViT-L-14-quickgelu', 'dfn2b'),
     ('ViT-L-14-336-quickgelu', 'openai'),
     ('ViT-H-14-quickgelu', 'metaclip_fullcc'),
     ('ViT-H-14-quickgelu', 'dfn5b'),
     ('ViT-H-14-378-quickgelu', 'dfn5b'),
     ('ViT-bigG-14-quickgelu', 'metaclip_fullcc')]



Below, I test a larger but more performant model based on the table ([here](https://github.com/mlfoundations/open_clip)):
```
model_name = "ViT-g-14"
checkpoint = "laion2b_s34b_b88k"
```

But, you can also opt for a smaller, less performant model:
```
model_name = "ViT-B-32"
checkpoint = "laion2b_s34b_b79k"
```

The model `model_name`,`checkpoint`  are set in `langchain_experimental.open_clip.py`.

For text, use the same method `embed_documents` as with other embedding models.

For images, use `embed_image` and simply pass a list of uris for the images.


```python
import numpy as np
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image

# Image URIs
uri_dog = "./data/dog.png"
uri_house = "./data/house.png"

# Embe images or text

#clip_embd = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
clip_embd = OpenCLIPEmbeddings(model_name="ViT-H-14-378-quickgelu", checkpoint="dfn5b")
img_feat_dog = clip_embd.embed_image([uri_dog])
img_feat_house = clip_embd.embed_image([uri_house])
text_feat_dog = clip_embd.embed_documents(["dog"])
text_feat_house = clip_embd.embed_documents(["house"])
```


```python
print(len(img_feat_dog[0]))
print(len(text_feat_dog[0]))
```

    1024
    1024
    

## 간단하게 테스트하기

Let's reproduce results shown in the OpenClip Colab [here](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=tMc1AXzBlhzm).


```python
# !pip install scikit-image
```

    Collecting scikit-image
      Downloading scikit_image-0.25.2-cp311-cp311-win_amd64.whl.metadata (14 kB)
    Requirement already satisfied: numpy>=1.24 in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (from scikit-image) (2.3.2)
    Requirement already satisfied: scipy>=1.11.4 in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (from scikit-image) (1.16.1)
    Requirement already satisfied: networkx>=3.0 in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (from scikit-image) (3.5)
    Requirement already satisfied: pillow>=10.1 in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (from scikit-image) (10.4.0)
    Collecting imageio!=2.35.0,>=2.33 (from scikit-image)
      Downloading imageio-2.37.0-py3-none-any.whl.metadata (5.2 kB)
    Collecting tifffile>=2022.8.12 (from scikit-image)
      Downloading tifffile-2025.8.28-py3-none-any.whl.metadata (32 kB)
    Requirement already satisfied: packaging>=21 in c:\users\sba\appdata\local\pypoetry\cache\virtualenvs\langchain-kr-us6bdj1p-py3.11\lib\site-packages (from scikit-image) (24.2)
    Collecting lazy-loader>=0.4 (from scikit-image)
      Downloading lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
    Downloading scikit_image-0.25.2-cp311-cp311-win_amd64.whl (12.8 MB)
       ---------------------------------------- 0.0/12.8 MB ? eta -:--:--
       ---------------- ----------------------- 5.2/12.8 MB 53.0 MB/s eta 0:00:01
       -------------------- ------------------- 6.6/12.8 MB 14.9 MB/s eta 0:00:01
       --------------------------- ------------ 8.7/12.8 MB 13.1 MB/s eta 0:00:01
       ---------------------------------------- 12.8/12.8 MB 17.8 MB/s  0:00:00
    Downloading imageio-2.37.0-py3-none-any.whl (315 kB)
    Downloading lazy_loader-0.4-py3-none-any.whl (12 kB)
    Downloading tifffile-2025.8.28-py3-none-any.whl (231 kB)
    Installing collected packages: tifffile, lazy-loader, imageio, scikit-image
    
       ---------------------------------------- 0/4 [tifffile]
       -------------------- ------------------- 2/4 [imageio]
       -------------------- ------------------- 2/4 [imageio]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ------------------------------ --------- 3/4 [scikit-image]
       ---------------------------------------- 4/4 [scikit-image]
    
    Successfully installed imageio-2.37.0 lazy-loader-0.4 scikit-image-0.25.2 tifffile-2025.8.28
    


```python
import os
from collections import OrderedDict

import IPython.display
import matplotlib.pyplot as plt
import skimage

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# 이미지를 묘사하는 문장들을 생성해서 descriptions에 저장
# multimodal 모델을 사용하면 쉽게 만들 수 있다.
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer",
}

# 이미지와 Description을 화면에 출력
original_images = []
images = []
image_uris = []  # List to store image URIs
texts = []
plt.figure(figsize=(16, 5))

# Loop to display and prepare images and assemble URIs
for filename in [
    filename
    for filename in os.listdir(skimage.data_dir)
    if filename.endswith(".png") or filename.endswith(".jpg")
]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image_path = os.path.join(skimage.data_dir, filename)
    image = Image.open(image_path).convert("RGB")

    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(image)  # Origional code does preprocessing here
    texts.append(descriptions[name])
    image_uris.append(image_path)  # Add the image URI to the list

plt.tight_layout()
```


    
![png](01-open_clip_files/01-open_clip_11_0.png)
    


### 참고 : Langchain의 OpenCLIPEmbeddings문서 페이지
https://python.langchain.com/api_reference/experimental/open_clip/langchain_experimental.open_clip.open_clip.OpenCLIPEmbeddings.html

주요 클래스 메서드
- embed_query
- embed_image
- embed_documents

### 이미지와 Description의 상관관계를 시각화 하기


```python
# Instantiate your model
clip_embd = OpenCLIPEmbeddings()

# Embed images and text
img_features = clip_embd.embed_image(image_uris)
text_features = clip_embd.embed_documents(["This is " + desc for desc in texts])

# Convert the list of lists to numpy arrays for matrix operations
img_features_np = np.array(img_features)
text_features_np = np.array(text_features)

# Calculate similarity
similarity = np.matmul(text_features_np, img_features_np.T)

# Plot
count = len(descriptions)
plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
# plt.colorbar()
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features", size=20)
```


    open_clip_model.safetensors:   0%|          | 0.00/3.94G [00:00<?, ?B/s]


    c:\Users\SBA\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-Us6BDj1P-py3.11\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\SBA\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
    To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
      warnings.warn(message)
    




    Text(0.5, 1.0, 'Cosine similarity between text and image features')




    
![png](01-open_clip_files/01-open_clip_14_3.png)
    

