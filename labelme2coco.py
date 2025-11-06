import json
import os
import numpy as np
from typing import List, Dict, Any

def labelme_to_coco_keypoints(labelme_dir: str, output_path: str):
    """
    将LabelMe标注的关键点转换为COCO格式
    
    Args:
        labelme_dir: LabelMe JSON文件所在的目录
        output_path: 输出的COCO JSON文件路径
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # COCO数据集结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 定义类别
    categories = [
        {
            "id": 1,
            "name": "hand",
            "keypoints": [
                "wrist_0", "wrist_1",
                "thumb_mcp", "thumb_pip", "thumb_dip",
                "index_mcp_0", "index_mcp_1", "index_pip", "index_dip",
                "middle_mcp_0", "middle_mcp_1", "middle_pip", "middle_dip",
                "ring_mcp_0", "ring_mcp_1", "ring_pip", "ring_dip",
                "pinky_mcp_0", "pinky_mcp_1", "pinky_pip", "pinky_dip"
            ],
            "skeleton": [
                [0, 1], [1, 2], [2, 3],
                [3, 4], [2, 5], [5, 6],
                [6, 7], [7, 8], [5, 9],
                [9, 10], [10, 11], [11, 12],
                [9, 13], [13, 14], [14, 15],
                [15, 16], [13, 17], [17, 18],
                [18, 19], [19, 20], [1, 17]
            ]
        }
    ]
    
    coco_data["categories"] = categories
    
    image_id = 1
    annotation_id = 1
    
    kpt_count = len(categories[0]["keypoints"])
    categories_names = [category["name"] for category in categories]
    
    # 遍历LabelMe JSON文件
    for filename in os.listdir(labelme_dir):
        if not filename.endswith('.json'):
            continue
            
        labelme_path = os.path.join(labelme_dir, filename)
        
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 添加图像信息
        image_info = {
            "id": image_id,
            "file_name": labelme_data["imagePath"],
            "width": labelme_data.get("imageWidth", 0),
            "height": labelme_data.get("imageHeight", 0)
        }
        coco_data["images"].append(image_info)
        
        kpts = [0] * kpt_count
        category_id = None
        num_keypoints = 0
        
        # 处理每个标注
        for shape in labelme_data["shapes"]:
            if shape["shape_type"] == "point":
                point = shape["points"]
                label = shape["label"]
                
                # 找到对应的类别ID
                try:
                    kpt_id = categories[0]["keypoints"].index(label)
                except:
                    print(f"unkonw name: {label} in " + filename)
                    exit(1)
                
                num_keypoints += 1
                # v=2: 已标注且可见, v=1: 已标注但不可见, v=0: 未标注
                kpts[kpt_id * 3:(kpt_id + 1) * 3] = [*point[0], 2]
                    
            elif shape["shape_type"] == "rectangle":
                points = shape["points"]
                label = shape["label"]
                try:
                    category_id = categories_names.index(label) + 1
                except:
                    print(f"unkonw category: {label} in " + filename)
                    exit(1)
                x_min, x_max = [points[0][0], points[1][0]] if points[0][0] <= points[1][0] else [points[1][0], points[0][0]]
                y_min, y_max = [points[0][1], points[1][1]] if points[0][1] <= points[1][1] else [points[1][1], points[0][1]]
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)
        
        if category_id is None:
            print("missing bbox in " + filename)
            exit(1)
        # 创建COCO标注
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "segmentation": [],  # 关键点检测通常不需要分割
            "keypoints": kpts,
            "num_keypoints": num_keypoints,
            "iscrowd": 0
        }
        
        coco_data["annotations"].append(annotation)
        annotation_id += 1
        image_id += 1
    
    # 保存COCO格式数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！共处理 {len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注")

# 使用示例
if __name__ == "__main__":
    labelme_to_coco_keypoints("/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/label", "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/coco_annotation/coco_keypoints.json")