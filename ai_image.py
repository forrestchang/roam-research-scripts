import json
import os
import re
import uuid
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


def get_children_blocks(block_id: str) -> List[str]:
    data = {
        "eid": f'[:block/uid "{block_id}"]',
        "selector": (
            "[:block/uid :node/title :block/string {:block/children [:block/uid :block/string]} {:block/refs [:node/title :block/string :block/uid]}]"
        ),
    }

    response = requests.post(f"{base_url}/pull", headers=headers, data=json.dumps(data))
    return response.json().get("result", {}).get(":block/children", [])


def write_new_block(block_id: str, type: str, content: str):
    if type == "caption":
        type_block = "Image Caption::"
    elif type == "ocr":
        type_block = "Image OCR::"

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


def generate_image_caption(image_url: str) -> str:
    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
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
    image_block_id = image_block[0]
    image_block_content = image_block[1]
    print(f"\nProcessing block: {image_block_id}")
    print(f"Block content: {image_block_content[:50]}...")  # Print first 50 characters

    children_blocks = get_children_blocks(image_block_id)
    print(f"Number of child blocks: {len(children_blocks)}")

    has_caption = False
    has_ocr = False

    for child in children_blocks:
        if child[":block/string"].startswith("Image Caption::"):
            has_caption = True
        elif child[":block/string"].startswith("Image OCR::"):
            has_ocr = True

    print(f"Existing caption: {has_caption}, Existing OCR: {has_ocr}")

    # If both caption and OCR exist, skip processing
    if has_caption and has_ocr:
        print("Both caption and OCR already exist. Skipping processing.")
        return

    image_urls = fetch_image_urls(image_block_content)
    print(f"Number of images in block: {len(image_urls)}")

    image_ocr_results = []
    image_captions = []
    for i, image_url in enumerate(image_urls, 1):
        print(f"\nProcessing image {i}/{len(image_urls)}")
        if is_valid_image(image_url):
            if not has_ocr:
                print("Generating OCR result...")
                ocr_result = generate_image_ocr_result(image_url)
                image_ocr_results.append(ocr_result)
                print(f"OCR result (first 50 chars): {ocr_result[:50]}...")

            if not has_caption:
                print("Generating image caption...")
                caption = generate_image_caption(image_url)
                image_captions.append(caption)
                print(f"Caption (first 50 chars): {caption[:50]}...")
        else:
            print(f"Skipping invalid image: {image_url}")

    if not has_ocr:
        print("Adding new OCR block...")
        ocr = "\n\n".join(image_ocr_results)
        write_new_block(image_block_id, "ocr", ocr)
    else:
        print("OCR already exists. Skipping.")

    if not has_caption:
        print("Adding new caption block...")
        caption = "\n\n".join(image_captions)
        write_new_block(image_block_id, "caption", caption)
    else:
        print("Caption already exists. Skipping.")

    print("Block processing complete.")


def load_processed_blocks():
    cache_file = Path("processed_blocks_cache.json")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return set(json.load(f))
    return set()


def save_processed_blocks(processed_blocks):
    cache_file = Path("processed_blocks_cache.json")
    with open(cache_file, "w") as f:
        json.dump(list(processed_blocks), f)


# Main execution
print("Starting image block processing...")
image_blocks = get_all_image_blocks()
print(f"Total number of image blocks: {len(image_blocks)}")

processed_blocks = load_processed_blocks()
print(f"Number of previously processed blocks: {len(processed_blocks)}")

# Wrap the loop with tqdm for a progress bar
for block in tqdm(image_blocks, desc="Processing image blocks", unit="block"):
    block_id = block[0]
    if block_id not in processed_blocks:
        add_image_caption_and_ocr_result(block)
        processed_blocks.add(block_id)
    else:
        print(f"Skipping already processed block: {block_id}")

save_processed_blocks(processed_blocks)

print("All image blocks processed. Script execution complete.")
