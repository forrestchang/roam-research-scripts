import functools
import hashlib
import json
import os
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from ocrmac import ocrmac
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

base_url = f"https://api.roamresearch.com/api/graph/{os.getenv('ROAM_GRAPH_NAME')}"
append_url = (
    f"https://append-api.roamresearch.com/api/graph/{os.getenv('ROAM_GRAPH_NAME')}"
)

headers = {
    "accept": "application/json",
    "X-Authorization": f"Bearer {os.getenv('ROAM_API_KEY')}",
    "Content-Type": "application/json",
}

CAPTION_PROMPT = """
Please carefully analyze this image and provide a detailed description, focusing on key information that could be used for future search and retrieval. 

Please summarize this information in a concise paragraph (about 3-8 sentences), ensuring to include all key words and descriptive phrases. Your description should be detailed enough to allow future retrieval of this image through various relevant keywords and concepts.

The result should be in English and Chinese.

The output format example:

**Summary (English)**
This is a detailed description of the image.

**Summary (Chinese)**
这是对图片的详细描述。
"""


@functools.lru_cache(maxsize=None)
def get_all_image_blocks():
    query = """
[:find ?block-uid ?block-str
 :in $ ?search-string
 :where
   [?b :block/uid ?block-uid]
   [?b :block/string ?block-str]
   [(clojure.string/includes? ?block-str ?search-string)]]
"""
    data = {
        "query": query,
        "args": ["![]("],
    }

    response = requests.post(f"{base_url}/q", headers=headers, data=json.dumps(data))

    return response.json().get("result", [])


def fetch_image_urls(block: str) -> List[str]:
    # Use regex to find all URLs within ![](...)
    urls = re.findall(r"!\[.*?\]\((.*?)\)", block)
    return urls


@functools.lru_cache(maxsize=100)
def get_children_blocks(block_id: str) -> List[str]:
    data = {
        "eid": f'[:block/uid "{block_id}"]',
        "selector": (
            "[:block/uid :node/title :block/string {:block/children [:block/uid :block/string]} {:block/refs [:node/title :block/string :block/uid]}]"
        ),
    }

    response = requests.post(f"{base_url}/pull", headers=headers, data=json.dumps(data))
    return response.json().get("result", {}).get(":block/children", [])


def write_new_block(block_id: str, content: str):
    type_block = "Image Caption::"

    data = {
        "location": {
            "block": {"uid": block_id},
            "nest-under": {"string": type_block},
        },
        "append-data": [{"string": content}],
    }

    response = requests.post(
        f"{append_url}/append-blocks", headers=headers, data=json.dumps(data)
    )
    return response


def is_valid_image(image_url: str) -> bool:
    # Check if the URL is a Firebase Storage URL
    if "firebasestorage.googleapis.com" in image_url:
        # Check if the URL contains a file extension
        if "." in image_url.split("/")[-1]:
            # Extract the file extension
            file_extension = image_url.split(".")[-1].split("?")[0].lower()
            # List of valid image extensions
            valid_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"]
            return file_extension in valid_extensions
        else:
            # If no file extension, check for "alt=media" in the URL
            return "alt=media" in image_url
    else:
        # For non-Firebase URLs, use the previous extension checking method
        valid_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
        return any(image_url.lower().endswith(ext) for ext in valid_extensions)


def generate_image_ocr_result(image_url: str) -> str:

    # Create a cache folder if it doesn't exist
    cache_folder = Path("image_cache")
    cache_folder.mkdir(exist_ok=True)

    # Parse the URL and get the file name
    parsed_url = urlparse(image_url)
    file_name = os.path.basename(parsed_url.path)

    # If file_name is empty (no extension), use the last part of the path
    if not file_name:
        file_name = parsed_url.path.split("/")[-1]

    # If still empty or no extension, generate a random name with .jpg extension
    if not file_name or "." not in file_name:
        random_name = str(uuid.uuid4())
        file_name = f"{random_name}.jpg"

    # Create the full path for the cached image
    cached_image_path = cache_folder / file_name

    # Download the image if it's not already in the cache
    if not cached_image_path.exists():
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(cached_image_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download image: HTTP {response.status_code}")

    print(cached_image_path)
    annotations = ocrmac.OCR(
        str(cached_image_path), language_preference=["zh-Hans", "en-US"]
    ).recognize()
    ocr_result = " ".join([e[0] for e in annotations])
    return ocr_result


def generate_image_caption(image_url: str, image_context: str, image_ocr: str) -> str:
    PROMPT = f"""
    请帮我按照下列需求来处理图片的内容：

    1. 我会提供图片中 OCR 的文本，在 <OCR_TEXT></OCR_TEXT> 中，图片的语境上下文会补充在 <IMAGE_CONTEXT></IMAGE_CONTEXT> 中
    2. 首先格式化 OCR 中的文本，如果 OCR 内容有部分缺失，请按照上下文补全（但是不要补全错误的信息），如果是英文内容，翻译一份中文版本（同时保留英文原文），使用易于阅读的格式，使用 markdown 格式（不使用标题）
    3. 根据图片 & OCR 文本解释图片
    4. 生成便于未来检索的关键词，使用中文 & 英文，多个关键词中使用 `,` 分隔，对于专有名词，不需要翻译成中文。关键词例子：Python, LLM (Large Language Model, 大预言模型)

    <OCR_TEXT>
    {image_ocr}
    </OCR_TEXT>

    <IMAGE_CONTEXT>
    {image_context}
    </IMAGE_CONTEXT>

    最终的输出格式：
    **OCR Result**

    **Image Explanation**

    **Keywords**
    """
    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content


def add_image_caption_and_ocr_result(image_block: list):
    block_id = image_block[0]
    image_block_content = image_block[1]
    print(f"\nProcessing block: {block_id}")
    print(f"Block content: {image_block_content[:50]}...")  # Print first 50 characters

    children_blocks = get_children_blocks(block_id)
    print(f"Number of child blocks: {len(children_blocks)}")

    has_caption = False

    for child in children_blocks:
        if child[":block/string"].startswith("Image Caption::"):
            has_caption = True

    print(f"Existing caption: {has_caption}")

    # If both caption and OCR exist, skip processing
    if has_caption:
        print("Caption already exists. Skipping processing.")
        return

    image_urls = fetch_image_urls(image_block_content)
    print(f"Number of images in block: {len(image_urls)}")

    image_captions = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(process_image, url, image_block_content): url
            for url in image_urls
        }
        for future in as_completed(future_to_url):
            caption = future.result()
            if caption:
                image_captions.append(caption)

    print("Adding new caption block...")
    caption = "\n\n".join(image_captions)
    write_new_block(block_id, caption)

    print("Block processing complete.")


def process_image(image_url, image_block_content):
    if is_valid_image(image_url):
        ocr_result = generate_image_ocr_result(image_url)
        caption = generate_image_caption(image_url, image_block_content, ocr_result)
        return caption
    return None


def init_db():
    conn = sqlite3.connect("processed_blocks.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS processed_blocks
                 (block_id TEXT PRIMARY KEY)"""
    )
    conn.commit()
    return conn


def is_block_processed(conn, block_id):
    c = conn.cursor()
    c.execute("SELECT 1 FROM processed_blocks WHERE block_id = ?", (block_id,))
    return c.fetchone() is not None


def mark_block_processed(conn, block_id):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO processed_blocks (block_id) VALUES (?)", (block_id,)
    )
    conn.commit()


# Main execution
print("Starting image block processing...")
image_blocks = get_all_image_blocks()
print(f"Total number of image blocks: {len(image_blocks)}")

conn = init_db()
for block in tqdm(image_blocks, desc="Processing image blocks", unit="block"):
    block_id = block[0]
    if not is_block_processed(conn, block_id):
        add_image_caption_and_ocr_result(block)
        mark_block_processed(conn, block_id)
    else:
        print(f"Skipping already processed block: {block_id}")

conn.close()

print("All image blocks processed. Script execution complete.")
