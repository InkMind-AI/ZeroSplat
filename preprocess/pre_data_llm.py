import json
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

def data_process_llm(
    json_path="",
    json_names=[],
    MODEL_NAME="",
    frame_choice_path="",
    save_json_path="",
    vis_save_root="",
    dataset_type="",
    api_key="",
    base_url="",
):
    """Run VLM referring localization via OpenAI-compatible API (e.g. DashScope)."""
    from openai import OpenAI

    from preprocess.qwen3vl8_image_text_answer_v2 import identify_category_and_locate
    from preprocess.read_reftext import batch_process_json

    if api_key is None or (isinstance(api_key, str) and api_key.strip() == ""):
        raise SystemExit("[VLM] Missing api_key (use EMPTY for local servers).")
    key = api_key
    if not base_url or not str(base_url).strip():
        raise SystemExit("[VLM] Missing base_url.")
    url = base_url
    if not MODEL_NAME or not str(MODEL_NAME).strip():
        raise SystemExit("[VLM] Missing MODEL_NAME / --model_name.")
    model_name = MODEL_NAME

    print(f"[VLM] API base_url={url}, model={model_name}")
    client = OpenAI(api_key=key, base_url=url)

    data_dict = batch_process_json(dataset_type, json_path, json_names)
    prompt_set = set()

    for file_content in data_dict.values():
        for sentence_list in file_content.values():
            for s in sentence_list:
                if s:
                    prompt_set.add(s)
                else:
                    print("This sentence is null")
                    sys.exit(1)

    category_results, localization_results = identify_category_and_locate(
        client, model_name, list(prompt_set), frame_choice_path, vis_save_root
    )

    try:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(category_results, f, ensure_ascii=False, indent=4)
        print(f"Category results saved to: {save_json_path}")
    except Exception as e:
        print(f"Failed to save category JSON: {e}")

    loc_save_path = save_json_path.replace(".json", "_bbox.json")
    try:
        with open(loc_save_path, "w", encoding="utf-8") as f:
            json.dump(localization_results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to save localization JSON: {e}")

    return category_results
