import json
import os
import yaml
from pathlib import Path
import numpy as np

def labelme_to_yolo_pose(labelme_json_path, output_yaml_path, kpt_idx:dict, class_names:dict):
    # 准备YOLO-Pose格式数据
    kpt_len = len(kpt_idx.keys())
    yolo_data = {
        'path': labelme_json_path,
        'train': 'images/train',  # 训练图像路径
        'val': 'images/val',      # 验证图像路径
        # 'test': 'images/test',    # 测试图像路径
        'kpt_shape': [kpt_len, 3],      # 关键点形状 [关键点数量, 维度(2 for x,y)]
        # 'flip_idx': [],           # 水平翻转时对应的关键点索引
        'names': class_names,     # 类别名称
        "kpt_names": {list(class_names.keys())[0]:list(kpt_idx.keys())}
    }
    
    # 自定义YAML输出以确保kpt_shape使用方括号格式
    class CustomDumper(yaml.Dumper):
        def represent_list(self, data):
            # 对于kpt_shape这样的短列表，使用流格式（方括号）
            if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
                return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            # 对于长列表，使用块格式
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    # 注册自定义表示器
    yaml.add_representer(list, CustomDumper.represent_list, Dumper=CustomDumper)
    
    # 写入YAML文件
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yolo_data, f, Dumper=CustomDumper, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"YOLO-Pose配置文件已生成: {output_yaml_path}")
    print(f"类别: {class_names}")
    
def find_key(dic:dict, val):
    keys_list = []
    for key, value in dic.items():
        if val == value:
            keys_list.append(key)
    return keys_list

def convert_labelme_annotations(labelme_json_path, output_dir, class_names:dict, kpt_idx:dict):
    """
    转换Labelme标注文件为YOLO-Pose格式的txt文件
    
    Args:
        labelme_json_path: Labelme JSON文件路径或目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    
    if os.path.isdir(labelme_json_path):
        json_files = [f for f in os.listdir(labelme_json_path) if f.endswith('.json')]
        json_paths = [os.path.join(labelme_json_path, f) for f in json_files]
    else:
        json_paths = [labelme_json_path]
    
    os.makedirs(output_dir, exist_ok=True)
    
    kpt_count = len(kpt_idx.keys())
    
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 准备YOLO格式的标注内容
        rect_array = np.zeros(5)
        kpt_array = np.zeros((kpt_count, 3))
        
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'point':
                # 关键点标注
                points = shape['points'][0]  # 第一个点
                x = points[0] / image_width
                y = points[1] / image_height
                
                # YOLO-Pose格式: class x y w h kpt1_x kpt1_y kpt2_x kpt2_y ...
                now_pt_class = shape['label']
                kpt_array[kpt_idx[now_pt_class], :] = [x, y, 1]
            elif shape['shape_type'] == 'rectangle':
                pt1 = shape['points'][0]
                pt2 = shape['points'][1]
                x = (pt1[0] + pt2[0]) / 2 / image_width
                y = (pt1[1] + pt2[1]) / 2 / image_height
                w = (pt2[0] - pt1[0]) / image_width
                h = (pt2[1] - pt1[1]) / image_height
                
                now_class = shape['label']
                class_idx = int(find_key(class_names, now_class)[0])
                rect_array[:] = [class_idx, x, y, w, h]
        
        rect_list = rect_array.tolist()
        kpt_list = kpt_array.flatten().tolist()
        yolo_lines = rect_list + kpt_list
        yolo_lines = ' '.join(map(str, yolo_lines))
        
        # 写入txt文件
        output_txt_path = os.path.join(output_dir, 
                                     os.path.splitext(os.path.basename(json_path))[0] + '.txt')
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(yolo_lines)
    
    print(f"标注文件已转换到: {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    LABELME_JSON_PATH = "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/label"  # Labelme JSON文件或目录
    OUTPUT_YAML_PATH = "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/hand.yaml"  # 输出的YAML配置文件路径
    OUTPUT_ANNOTATIONS_DIR = "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/yolo_label"  # 转换后的标注文件输出目录
    KPT_IDX = {
        "wrist_0":0,
        "wrist_1":1,
        "thumb_mcp":2,
        "thumb_pip":3,
        "thumb_dip":4,
        "index_mcp_0":5,
        "index_mcp_1":6,
        "index_pip":7,
        "index_dip":8,
        "middle_mcp_0":9,
        "middle_mcp_1":10,
        "middle_pip":11,
        "middle_dip":12,
        "ring_mcp_0":13,
        "ring_mcp_1":14,
        "ring_pip":15,
        "ring_dip":16,
        "pinky_mcp_0":17,
        "pinky_mcp_1":18,
        "pinky_pip":19,
        "pinky_dip":20
    }
    TARGET_NAME = {0:"hand"}
    
    # # 定义类别名称（根据你的实际类别修改）
    # CLASS_NAMES = ["person", "face", "hand"]  # 示例类别
    
    # # 定义关键点连接关系（可选）
    # SKELETON = [
    #     [0, 1], [1, 2], [2, 3],  # 示例骨架连接
    #     [0, 4], [4, 5], [5, 6]   # 根据你的关键点定义修改
    # ]
    
    # 生成YAML配置文件
    labelme_to_yolo_pose(LABELME_JSON_PATH, OUTPUT_YAML_PATH, KPT_IDX, TARGET_NAME)
    
    # 转换标注文件（可选）
    convert_labelme_annotations(LABELME_JSON_PATH, OUTPUT_ANNOTATIONS_DIR, TARGET_NAME, KPT_IDX)