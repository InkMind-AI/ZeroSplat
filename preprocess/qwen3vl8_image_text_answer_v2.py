import json
import os
import base64
import sys
import re
import cv2
from openai import APITimeoutError

def parse_json_safely(text):
    if not text: return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    text_stripped = text.strip()

    while idx < len(text_stripped):
        text_stripped = text_stripped[idx:].lstrip()
        if not text_stripped:
            break
        try:
            obj, end_idx = decoder.raw_decode(text_stripped)
            objects.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            break

    if len(objects) > 1:
        return objects
    elif len(objects) == 1:
        return objects[0]

    return None

def _get_box_coords(obj):
    b = obj.get("box") or obj.get("bbox_2d") or obj.get("bbox")
    if isinstance(b, list) and len(b) == 4:
        return b
    return None

def _normalize_parsed_to_boxes_input(parsed):
    if not isinstance(parsed, dict):
        return parsed
    multi = parsed.get("MULTIPLE objects") or parsed.get("MULTIPLE object")
    single = parsed.get("SINGLE object") or parsed.get("SINGLE objects")
    if isinstance(multi, list) and len(multi) > 0:
        return multi
    if isinstance(single, dict) and _get_box_coords(single) is not None:
        return [single]
    return parsed

def extract_boxes_from_response(parsed):
    if parsed is None:
        return []
    parsed = _normalize_parsed_to_boxes_input(parsed)
    boxes = []
    if isinstance(parsed, dict):
        b = _get_box_coords(parsed)
        if b is not None:
            s = parsed.get("score", 0.0)
            lbl = parsed.get("label", "")
            boxes.append((b, s, lbl))
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                b = _get_box_coords(item)
                if b is not None:
                    s = item.get("score", 0.0)
                    lbl = item.get("label", "")
                    boxes.append((b, s, lbl))
    return boxes

def encode_image(image_path):
    if not os.path.exists(image_path):
        print("Image does not exist")
        sys.exit(1)
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def identify_category_and_locate(client, model_name, prompt_text_list, frame_choice_path,vis_save_root):

    print(f"Reading image list from {frame_choice_path}")
    with open(frame_choice_path, 'r') as f:
        data = json.load(f)

    frame_files = data[0].get("frame_files", [])
    top_frames = frame_files[:]
    print(f"Selected top {len(top_frames)} images.")

    image_content_list = []
    valid_image_info = []

    for idx, img_path in enumerate(top_frames):
        if os.path.exists(img_path):
            img_cv = cv2.imread(img_path)
            if img_cv is not None:
                h, w = img_cv.shape[:2]
                b64_str = encode_image(img_path)
                current_img_idx = len(valid_image_info)
                valid_image_info.append({
                    "path": img_path,
                    "w": w,
                    "h": h,
                    "idx": current_img_idx,
                    "original_idx": idx,
                    "b64": b64_str
                })

                image_content_list.append({"type": "text", "text": f"Image {current_img_idx + 1}:"})
                image_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})

    if not image_content_list:
        print("No valid images")
        sys.exit(1)

    print(f" [Phase 1] Running semantic categorization...")

    input_str = "\n".join([f"- {text}" for text in prompt_text_list])

    system_instruction_cat = (
    "You are an expert in visual semantic conversion and scene understanding. "
    "I will provide images and a list of 'Referring Expressions' (long descriptions). "
    "Your task is to identify ALL object categories mentioned in each expression. "
    "An expression may refer to a single object or multiple distinct objects; you must make a comprehensive judgment independently based on the context and ALL provided images.\n\n"
    "STRICT RULES:\n"
    "1. Output valid JSON: {\"original expression\": [\"category1\", \"category2\", ...]}.\n"
    "2. SPATIAL & CONTEXTUAL TARGETING: Use spatial relationships (e.g., 'in front of', 'next to', 'on the top') to locate the specific target object(s) in the image. You must analyze the image(s) to ensure the identified object is the one intended by the text.\n"
    "3. ABSTRACTION (Color + Noun): Convert the identified target object into a simplified 'Color + Noun' phrase based on its actual appearance in the images.\n"
    "   Examples:\n"
    "   - Input: 'A round object with a smooth surface directly in front of the white doll' -> If the object in the image is a blue cup, Output: ['blue cup']\n"
    "   - Input: 'fire alarm and smoke detector' -> Output: ['red fire alarm', 'white smoke detector']\n"
    "   - Input: 'the seating furniture on the left' -> If it's a brown chair, Output: ['brown chair']\n"
    "4. QUANTITY: If one expression refers to multiple objects, return them all in the array.\n"
    "5. Do NOT output markdown formatting (```json), just the raw JSON string.\n"
    "6. COMPLETENESS: Return a mapping for EVERY expression in the input list.\n"
    )

    user_text_cat = (
        f"Here is the list of descriptions to convert:\n"
        f"{input_str}\n\n"
        f"Please map each description to its specific object category noun:"
    )

    messages_cat = [
        {"role": "system", "content": system_instruction_cat},
        {"role": "user", "content": image_content_list + [{"type": "text", "text": user_text_cat}]}
    ]

    category_results = {}
    raw_result = ""
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        retry_count += 1
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages_cat,
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            print(
                f"Phase 1 API error (attempt {retry_count}/{max_retries}): {e}"
            )
            continue
        choice = completion.choices[0]
        raw_result = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            print(
                f"Phase 1 response truncated (finish_reason=length), "
                f"attempt {retry_count}/{max_retries}, retrying..."
            )
            continue
        if not raw_result:
            print(
                f"Empty Phase 1 response (attempt {retry_count}/{max_retries}), retrying..."
            )
            continue
        if raw_result.startswith("```json"):
            raw_result = raw_result.replace("```json", "").replace("```", "").strip()
        elif raw_result.startswith("```"):
            raw_result = raw_result.replace("```", "").strip()
        parsed_json = parse_json_safely(raw_result)
        if not isinstance(parsed_json, dict):
            print(
                f"Phase 1 JSON parse failed (attempt {retry_count}/{max_retries}), "
                f"len={len(raw_result)}; retrying..."
            )
            continue
        category_results = {}
        for key, val in parsed_json.items():
            if isinstance(val, str):
                category_results[key] = [
                    str(val).lower().strip().replace(".", "").replace('"', "")
                ]
            elif isinstance(val, list):
                category_results[key] = [
                    str(v).lower().strip().replace(".", "").replace('"', "") for v in val
                ]
            else:
                category_results[key] = []
        print("Phase 1 categorization done.")
        break
    else:
        print(f"Failed Phase 1 categorization after {max_retries} attempts")
        sys.exit(1)

    if not category_results:
        print(f"Failed to query LLM after {max_retries} attempts")
        sys.exit(1)

    final_category_results = category_results
    all_categories = []
    for cat_list in category_results.values():
        if isinstance(cat_list, list):
            all_categories.extend(cat_list)
        else:
            all_categories.append(cat_list)
    unique_categories = list(set(all_categories))
    print(f"Identified categories : {unique_categories}")

    target_images_info = valid_image_info
    print(f"[Phase 2] Visual localization on key frames ({len(target_images_info)} images)...")

    system_instruction_loc = (
    "You are an expert in image localization and visual grounding. "
    "I will provide you with MULTIPLE images and a single referring text. "
    "Your task is to precisely locate ALL target objects described by the text within the images.\n\n"
    "CORE REASONING INSTRUCTIONS:\n"
    "1. VISUAL & SPATIAL CUES: Strictly rely on spatial relations, color descriptors, and object attributes. "
    "If the text describes 'A round object with a smooth surface directly in front of the white doll', "
    "you must first identify the white doll as an anchor and then locate the specific object(s) in front of it based on spatial context.\n"
    "2. CROSS-IMAGE SPATIAL REASONING: Analyze ALL provided images to maintain consistency. "
    "Use global spatial reasoning to determine if an object in the current frame is indeed part of the target group described by the text.\n"
    "3. ABSENCE OF TARGET: Only return -1 if the target object(s) are absolutely not present.\n\n"
    "IMPORTANT RULES:\n"
    "1. Coordinates must be normalized [0, 1000], order: [xmin, ymin, xmax, ymax]. Use the key \"box\" for the coordinates."
)
    localization_results = {key: {} for key in prompt_text_list}

    # Per-query timeout for Phase 2: a single (frame, text) query that exceeds this
    # is skipped (treated as "no detection") instead of aborting the whole pipeline.
    loc_timeout_s = float(os.environ.get("VLM_LOC_TIMEOUT_S", "180"))
    loc_client = client.with_options(timeout=loc_timeout_s, max_retries=0)
    total_queries = 0
    skipped_queries = []
    print(f"[Phase 2] per-query timeout={loc_timeout_s:.0f}s (set VLM_LOC_TIMEOUT_S to change)")

    VIS_SAVE_ROOT = vis_save_root
    os.makedirs(VIS_SAVE_ROOT, exist_ok=True)
    for img_idx, img_info in enumerate(target_images_info):
        img_path = img_info['path']
        img_name = os.path.basename(img_path)
        print(f"Analyzing image ({img_idx+1}/{len(target_images_info)}): {img_name} ...")

        raw_img = cv2.imread(img_path)

        for target_text in prompt_text_list:
            categories = final_category_results.get(target_text, [target_text])
            if not isinstance(categories, list):
                categories = [categories]
            short_names = ", ".join(categories)

            prompt_loc = (
                f"Detect ALL instances of '{target_text}' (Categories: {short_names}) in this image.\n"
                "This text may refer to MULTIPLE object types. Detect ALL of them.\n"
                "Return format (use ONLY these exact keys, output raw JSON only):\n"
                "- MULTIPLE objects: [{{\"box\": [xmin, ymin, xmax, ymax], \"score\": , \"label\": \"object1\"}}, ...]\n"
                "- SINGLE object: {{\"box\": [xmin, ymin, xmax, ymax], \"score\": , \"label\": \"object\"}}\n"
                "RULES:\n"
                "1. Each object MUST use key \"box\" (NOT bbox, NOT bbox_2d) for coordinates.\n"
                "2. Coordinates normalized [0, 1000], order: [xmin, ymin, xmax, ymax]\n"
                "3. Use structural boundaries to distinguish objects\n"
                "4. Return -1 ONLY if absolutely not present\n"
                "5. 'label' field must describe each detected instance"
                )

            total_queries += 1
            try:
                response = loc_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_instruction_loc},
                            {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_info['b64']}"}},
                                    {"type": "text", "text": prompt_loc}
                                ]
                            }
                        ],
                        temperature=0.1
                    )
            except APITimeoutError:
                skipped_queries.append((img_name, target_text))
                print(
                    f"  [SKIP] timeout (>{loc_timeout_s:.0f}s) frame={img_name} "
                    f"text={target_text[:80]!r}; continue with next text"
                )
                continue
            content = (response.choices[0].message.content or "").strip()
            print(f"  [LLM response] {content[:200]}..." if len(content) > 200 else f"  [LLM response] {content}")

            parsed = parse_json_safely(content)
            boxes_with_info = extract_boxes_from_response(parsed)

            if not boxes_with_info:
                continue

            final_boxes = []
            final_scores = []
            final_labels = []

            for found_box, confidence, label in boxes_with_info:
                v1, v2, v3, v4 = found_box

                if v1 == -1 or sum(found_box) < 0:
                    continue

                if max(v1, v2, v3, v4) <= 1.0 and max(v1, v2, v3, v4) > 0:
                    v1, v2, v3, v4 = v1*1000, v2*1000, v3*1000, v4*1000

                # v1=xmin, v2=ymin, v3=xmax, v4=ymax
                x_min = int(v1 / 1000 * img_info['w'])
                y_min = int(v2 / 1000 * img_info['h'])
                x_max = int(v3 / 1000 * img_info['w'])
                y_max = int(v4 / 1000 * img_info['h'])

                final_box = [
                    max(0, min(x_min, img_info['w'])),
                    max(0, min(y_min, img_info['h'])),
                    max(0, min(x_max, img_info['w'])),
                    max(0, min(y_max, img_info['h']))
                ]

                box_area = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
                if box_area <= 0:
                    continue

                final_boxes.append(final_box)
                final_scores.append(confidence)
                final_labels.append(label if label else f"{categories[0] if categories else 'obj'}_{len(final_boxes)}")

            if not final_boxes:
                continue

            localization_results[target_text][img_name] = {
                "box": final_boxes[0] if len(final_boxes) == 1 else final_boxes,
                "score": final_scores[0] if len(final_scores) == 1 else final_scores,
                "label": final_labels[0] if len(final_labels) == 1 else final_labels
            }

            if raw_img is not None:
                vis_img = raw_img.copy()
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

                for i, (box, score, lbl) in enumerate(zip(final_boxes, final_scores, final_labels)):
                    color = colors[i % len(colors)]
                    cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label_text = f"{lbl} | {score:.2f}"
                    cv2.putText(vis_img, label_text, (box[0], box[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                safe_key_name = re.sub(r'[\\/*?:"<>|]', "", target_text).replace(" ", "_")[:50]
                desc_save_dir = os.path.join(VIS_SAVE_ROOT, safe_key_name)
                os.makedirs(desc_save_dir, exist_ok=True)

                save_filename = f"multi_{len(final_boxes)}_vis_{img_name}"
                save_path = os.path.join(desc_save_dir, save_filename)
                cv2.imwrite(save_path, vis_img)
                print(f"  Detected {len(final_boxes)} object(s), visualization saved")
    print(f"[Phase 2] skipped {len(skipped_queries)}/{total_queries} queries due to timeout")
    for sk_img, sk_text in skipped_queries:
        print(f"  [SKIPPED] frame={sk_img} text={sk_text[:80]!r}")
    json_save_path = os.path.join(VIS_SAVE_ROOT, "localization_results.json")
    with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(localization_results, f, indent=4, ensure_ascii=False)
    print(f"Final localization results saved to: {json_save_path}")

    return final_category_results, localization_results
