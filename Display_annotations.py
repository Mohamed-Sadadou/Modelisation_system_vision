import json
import os
import cv2
from collections import defaultdict

def draw_bounding_boxes(json_file, images_folder, output_folder):
    with open(json_file, 'r') as file:
        annotations = json.load(file)

    image_annotations = defaultdict(list)
    for annotation in annotations:
        image_annotations[annotation['path_name']].append(annotation)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_path, annotations in image_annotations.items():
        full_image_path = os.path.join(images_folder, image_path)
        
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Error: Unable to open image {full_image_path}")
            continue

        for annotation in annotations:
            label = annotation['label']
            coordinates = annotation['coordinates']
            color = annotation.get('color', [255, 0, 0])  
            bgr_color = tuple(reversed(color))

            x, y, w, h = coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        output_path = os.path.join(output_folder, image_path)
        cv2.imwrite(output_path, image)

json_file = 'annotations_test.json'
images_folder = 'Test'
output_folder = 'output_images'

draw_bounding_boxes(json_file, images_folder, output_folder)
