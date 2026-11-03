import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from preprocess.read_reftext import batch_process_json

def dataset_dir_name(dataset_type: str) -> str:
    """Folder name under base_data_dir (CLI type may differ from on-disk name)."""
    if dataset_type == "gr_scannet":
        return "gr-scannet"
    return dataset_type

def select_test_image(dataset_type,dataset_name):
    if dataset_type in ["Ref-lerf","grlerf"]:
        scene_gt_frames = {
        "waldo_kitchen": ['frame_00019', 'frame_00035', 'frame_00067', 'frame_00105','frame_00162'],
        "ramen": ['frame_00006','frame_00024','frame_00060','frame_00065','frame_00081','frame_00119','frame_00128'],
        "figurines": ['frame_00083', 'frame_00097', 'frame_00146', 'frame_00179'],
        "teatime": ['frame_00002','frame_00025','frame_00043','frame_00107','frame_00129','frame_00140'],
        "5b186c5a-2": ['00000', '00026', '00054']
        }
        frame_names = scene_gt_frames[dataset_name]
    elif dataset_type == "lerf":
        scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
        }
        frame_names = scene_gt_frames[dataset_name]
    elif dataset_type in ["scannet","gr_scannet"]:
        frame_names = [""]
    else:
        print("Please provide a valid dataset type")
        sys.exit(1)
    return frame_names

def get_prompt_render(dataset_type,dataset_name, base_data_dir=None):
    if not base_data_dir:
        base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if dataset_type in ["Ref-lerf","grlerf"]:
        data_target_name = {
        "waldo_kitchen": ['frame_00019.json', 'frame_00035.json', 'frame_00067.json', 'frame_00105.json','frame_00162.json'],
        "ramen": ['frame_00006.json','frame_00024.json','frame_00060.json','frame_00065.json','frame_00081.json','frame_00119.json','frame_00128.json'],
        "figurines": ['frame_00083.json', 'frame_00097.json', 'frame_00146.json', 'frame_00179.json'],
        "teatime": ['frame_00002.json','frame_00025.json','frame_00043.json','frame_00107.json','frame_00129.json','frame_00140.json'],
        }
        json_names = data_target_name[dataset_name]
        if dataset_type == "Ref-lerf":
            json_path = os.path.join(base_data_dir, "Ref-lerf", dataset_name, "json", "test_json")
        if dataset_type == "grlerf":
            json_path = os.path.join(base_data_dir, "grlerf", dataset_name, "json")
        data_dict = batch_process_json(dataset_type, json_path, json_names)
        prompt_set = set()
        for file_content in data_dict.values():
            for sentence_list in file_content.values():
                if not isinstance(sentence_list, (list, tuple)):
                    sentence_list = [sentence_list] if sentence_list else []
                for s in sentence_list:
                    if s:
                        prompt_set.add(s)
                    else:
                        print("This sentence is null")
                        sys.exit(1)
        target_name = list(prompt_set)
        print(target_name)
    elif dataset_type in ["gr_scannet"]:
        data_target_name =  {
            "scene0070_00":["scene0070_00.json"],
            "scene0000_00":["scene0000_00.json"],
            "scene0062_00":["scene0062_00.json"],
            "scene0097_00":["scene0097_00.json"],
            "scene0140_00":["scene0140_00.json"],
            "scene0200_00":["scene0200_00.json"],
            "scene0347_00":["scene0347_00.json"],
            "scene0400_00":["scene0400_00.json"],
            "scene0590_00":["scene0590_00.json"],
            "scene0645_00":["scene0645_00.json"]
            }
        json_names = data_target_name[dataset_name]
        json_path = os.path.join(
            base_data_dir, dataset_dir_name(dataset_type), dataset_name, f"{dataset_name}_text"
        )
        data_dict = batch_process_json(dataset_type, json_path, json_names)
        prompt_set = set()
        
        for file_content in data_dict.values():
            for sentence_list in file_content.values():
                if not isinstance(sentence_list, (list, tuple)):
                    sentence_list = [sentence_list] if sentence_list else []
                for s in sentence_list:
                    if s:
                        prompt_set.add(s)
                    else:
                        print("This sentence is null")
                        sys.exit(1)
        target_name = list(prompt_set)
        print(target_name)

    elif dataset_type == "lerf":
        data_target_name =  {
            "waldo_kitchen":['knife', 'plate', 'spatula',  'yellow desk', 'spoon', 'dark cup', 'pour-over vessel', 'plastic ladle', 'toaster', 'refrigerator', 'ottolenghi', 'frog cup', 'Stainless steel pots', 'sink', 'pot', 'red cup', 'ketchup', 'cabinet'],
            "figurines":['miffy', 'toy elephant', 'green apple', 'spatula', 'pirate hat', 'tesla door handle', 'porcelain hand', 'toy cat statue', 'pumpkin', 'rubics cube', 'jake',  'pink ice cream', 'rubber duck with buoy', 'red toy chair', 'bag', 'red apple', 'rubber duck with hat', 'waldo', 'pikachu', 'old camera', 'green toy chair'],
            "ramen":['nori', 'plate', 'corn', 'sake cup', 'bowl', 'chopsticks', 'hand', 'spoon', 'wavy noodles', 'egg', 'kamaboko', 'onion segments', 'napkin', 'glass of water'],
            "teatime":['sheep', 'bear nose', 'tea in a glass', 'bag of cookies', 'three cookies', 'hooves', 'coffee', 'paper napkin', 'dall-e brand', 'yellow pouf', 'plate', 'apple', 'coffee mug', 'stuffed bear']
            }
        target_name = data_target_name[dataset_name]
    elif dataset_type == "scannet":
        data_target_name =  {
            "scene0070_00":["cabinet", "bed", "chair","window", "picture", "curtain","table","floor","wall"],
            "scene0000_00":["wall", "floor", "cabinet","bed", "sofa", "table","door","window","counter","desk","curtain","refrigerator","toilet","sink"],
            "scene0062_00":["wall", "floor","door","counter","toilet","sink"],
            "scene0097_00":["wall","floor","cabinet","door","counter","refrigerator","sink"],
            "scene0140_00":["wall","floor","cabinet","chair","table","door","window","picture"],
            "scene0200_00":["wall","floor","cabinet","bed","chair","door","window","desk"],
            "scene0347_00":["wall","floor","cabinet","door","counter","refrigerator","sink"],
            "scene0400_00":["wall","floor","bed","chair","door","window","desk"],
            "scene0590_00":["wall", "floor", "cabinet","bed","chair", "sofa", "table","door","picture","counter","desk","curtain","refrigerator","sink"],
            "scene0645_00":["wall", "floor", "cabinet","bed","chair", "sofa", "table","door","window","picture","desk","curtain","shower curtain","toilet","sink","bathtub"]
            }
        target_name = data_target_name[dataset_name]
    else:
        print("Please provide a valid dataset type")
        sys.exit(1)
    return target_name
    
def get_json_names(dataset_type,dataset_name):
    if dataset_type in ["Ref-lerf","grlerf"]:
        scene_gt_frames = {
        "waldo_kitchen": ['frame_00019.json', 'frame_00035.json', 'frame_00067.json', 'frame_00105.json','frame_00162.json'],
        "ramen": ['frame_00006.json','frame_00024.json','frame_00060.json','frame_00065.json','frame_00081.json','frame_00119.json','frame_00128.json'],
        "figurines": ['frame_00083.json', 'frame_00097.json', 'frame_00146.json', 'frame_00179.json'],
        "teatime": ['frame_00002.json','frame_00025.json','frame_00043.json','frame_00107.json','frame_00129.json','frame_00140.json'],
        }
        json_names = scene_gt_frames[dataset_name]
    elif dataset_type == "gr_scannet":
        scene_gt_frames = {
            "scene0070_00":["scene0070_00.json"],
            "scene0000_00":["scene0000_00.json"],
            "scene0062_00":["scene0062_00.json"],
            "scene0097_00":["scene0097_00.json"],
            "scene0140_00":["scene0140_00.json"],
            "scene0200_00":["scene0200_00.json"],
            "scene0347_00":["scene0347_00.json"],
            "scene0400_00":["scene0400_00.json"],
            "scene0590_00":["scene0590_00.json"],
            "scene0645_00":["scene0645_00.json"]
            }
        json_names = scene_gt_frames[dataset_name]
    elif dataset_type == "lerf":
        scene_gt_frames = {
        "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
        "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
        "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
        "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
        }
        json_names = scene_gt_frames[dataset_name]
    elif dataset_type in ["scannet","gr_scannet"]:
        json_names = []
    else:
        print("Please provide a valid dataset type")
        sys.exit(1)
    return json_names

    
