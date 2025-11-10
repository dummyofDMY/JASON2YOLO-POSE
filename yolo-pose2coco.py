import os
import json
import cv2
from glob import glob
import time

def yolo_pose_to_coco(image_dir, label_dir, output_json, keypoint_names, category_name="person"):
    images = []
    annotations = []
    categories = [{
        "id": 1,
        "name": category_name,
        "supercategory": category_name,
        "keypoints": keypoint_names,
        "skeleton": []  # 可以根据需要定义骨架连接关系
    }]
    info = dict(url="whate can I say?",
                date_created=time.strftime("%a, %d %b %Y %H:%M:%S +0000",time.localtime()),
                description="Conversion of YOLO dataset into MS-COCO format")
    
    ann_id = 1
    img_id = 1

    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))

    for label_path in label_files:
        image_name = os.path.basename(label_path).replace(".txt", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            image_name = image_name.replace(".jpg", ".png")
            image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"⚠️ 找不到图像: {image_name}, 跳过")
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # 添加图像信息
        images.append({
            "id": img_id,
            "file_name": image_name,
            "width": w,
            "height": h
        })

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5 + len(keypoint_names)*3:
                print(f"⚠️ {label_path} 数据不完整，跳过")
                continue

            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])

            # 转换为像素坐标
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw *= w
            bh *= h

            keypoints = []
            num_keypoints = 0
            kp_values = parts[5:]

            for i in range(len(keypoint_names)):
                x_rel = float(kp_values[i*3])
                y_rel = float(kp_values[i*3 + 1])
                v = int(float(kp_values[i*3 + 2]))

                if v > 0:
                    num_keypoints += 1
                    keypoints.extend([x_rel * w, y_rel * h, v])
                else:
                    keypoints.extend([0, 0, 0])

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "num_keypoints": num_keypoints,
                "keypoints": keypoints
            })

            ann_id += 1
        img_id += 1

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info":info
    }

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)
    print(f"✅ 转换完成: {output_json}")


# === 示例使用 ===
if __name__ == "__main__":
    # YOLO-Pose 数据路径
    image_dir = "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/coco_annotation/train"
    label_dir = "/home/dummy/Documents/graduate_design/dataset/10_21_hand_keypoints2/hand_10_21/labels/train"

    # 定义关键点名称
    keypoint_names = [
        "wrist_0", "wrist_1",
        "thumb_mcp", "thumb_pip", "thumb_dip",
        "index_mcp_0", "index_mcp_1", "index_pip", "index_dip",
        "middle_mcp_0", "middle_mcp_1", "middle_pip", "middle_dip",
        "ring_mcp_0", "ring_mcp_1", "ring_pip", "ring_dip",
        "pinky_mcp_0", "pinky_mcp_1", "pinky_pip", "pinky_dip"
    ]

    output_json = "coco_keypoints.json"

    yolo_pose_to_coco(image_dir, label_dir, output_json, keypoint_names)
