import cv2
import numpy as np
import os
import json

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

colors_hex = {
    "Jaune": ("#FCFF00", "Texte Illisible"),
    "Bleu": ("#001EFF", "Formes Confuses"),
    "Vert": ("#00FF00", "Hybrides Mi-Animal Mi-Homme"),
    "Orange": ("#FF6600", "Flou de Mouvement Probleme de Diffusion"),
    "Rouge": ("#FF0000", "Incoherences Spatiales")
}

colors_rgb = {}
for color, (hex_color, label) in colors_hex.items():
    rgb_color = hex_to_rgb(hex_color)
    print(f"hex_color {hex_color} -> rgb_color {rgb_color}")
    colors_rgb[color] = {
        "rgb": rgb_color,
        "label": label
    }

def extract_bounding_boxes(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erreur : impossible de charger l'image à partir de {image_path}")
        return [], None

    bounding_boxes = []

    for color, data in colors_rgb.items():
        rgb_color = data["rgb"]
        label = data["label"]
        bgr_color = tuple(reversed(rgb_color)) 
        thr = 30

        mask_r = cv2.inRange(image[:, :, 2], rgb_color[0] - thr, rgb_color[0] + thr)
        mask_g = cv2.inRange(image[:, :, 1], rgb_color[1] - thr, rgb_color[1] + thr)
        mask_b = cv2.inRange(image[:, :, 0], rgb_color[2] - thr, rgb_color[2] + thr)

        mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append({
                "label": label,
                "coordinates": (x, y, w, h),
                "color": rgb_color
            })

    if not bounding_boxes:
        cv2.putText(image, "No artifacts detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return bounding_boxes, image

def process_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    annotations = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            bounding_boxes, annotated_image = extract_bounding_boxes(image_path)

            if annotated_image is not None:
                annotated_image_path = os.path.join(output_folder, f"annotated_{filename}")
                cv2.imwrite(annotated_image_path, annotated_image)

            if not bounding_boxes:
                annotations.append({
                    "path_name": filename,
                    "label": "No artifacts detected",
                    "coordinates": None,
                    "color": None
                })
            else:
                for box in bounding_boxes:
                    annotations.append({
                        "path_name": filename,
                        "label":"artifact",
                        "coordinates": box['coordinates'],
                        "color": box['color']
                    })

    return annotations

folder_path = "./Test_cadre"
output_folder = "./annotated_test_images"
annotations = process_images_in_folder(folder_path, output_folder)

json_file = "./annotations_test.json"
with open(json_file, mode='w') as file:
    json.dump(annotations, file, indent=4)

print(f"Les annotations ont été sauvegardées dans {json_file}")
print(f"Les images annotées ont été sauvegardées dans le dossier {output_folder}")
