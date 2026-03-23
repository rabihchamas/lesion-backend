from ultralytics_lesions import YOLO
import cv2
import os
import platform
import torch
import torchvision.ops as ops
from app.scripts.cnn_preds import cnn_model_predict

  


def plot_boxes_on_image(image_path, boxes, predictions, default_color=(255, 0, 0), highlight_color=(0, 0, 255), thickness=2, threshold=0.255):
    img = cv2.imread(image_path)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if predictions is not None:
            pred = predictions[i]
            color = highlight_color if pred > threshold else default_color
            #label = f"{pred:.2f}"
            label = "Suspicious (model)" if pred > threshold else "Benign (model)"
            # Put prediction label
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            color = default_color


        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img



def detect_lesions(yolo_model, img_path, conf, out_conf=None, crop_size=6020//6, overlap=300, size=None):
    if size is None:
        size = crop_size
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # Calculate steps with overlap
    step_x = crop_size - overlap
    step_y = crop_size - overlap

    outputs=[]
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            x_end = min(x + crop_size, w)
            y_end = min(y + crop_size, h)
            crop = image[y:y_end, x:x_end]
            results = yolo_model(crop, imgsz=size, conf=conf)  # Predict on an image
            print(f"confidence: {results[0].boxes.conf}")
            boxes = results[0].boxes.xyxy.detach()
            boxes_list = boxes.cpu().int().tolist()
            # Adjust labels
            new_labels = []
            for line in boxes_list:
                cx, cy, bw, bh =  line
                new_labels.append([cx + x, cy+ y, bw + x, bh + y])
            outputs.extend(new_labels)
    if out_conf is not None:        
        # One zoom out shot for big lesions
        results = yolo_model(img_path, imgsz=size, conf=out_conf)
        boxes = results[0].boxes.xyxy.detach()
        boxes_list = boxes.cpu().int().tolist()
        if len(boxes_list)>0:
            outputs.extend(boxes_list)

    if len(outputs) > 0:
        boxes_tensor = torch.tensor(outputs, dtype=torch.float32)
        
        # Calculate areas of boxes (width * height)
        widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
        heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
        areas = widths * heights
        
        # Use areas as scores so bigger boxes have higher priority
        scores = areas
        
        # Apply NMS with a suitable IoU threshold
        keep_indices = ops.nms(boxes_tensor, scores, iou_threshold=0.1)
        filtered_boxes = boxes_tensor[keep_indices].int().tolist()
    else:
        filtered_boxes = []
    return filtered_boxes, image, img_path

def classify_and_plot_boxes(cnn_model, image, img_path, boxes):    
    predictions = []
    for (x1, y1, x2, y2) in boxes:
        crop = image[y1:y2, x1:x2]
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = cnn_model_predict(cnn_model, crop, device='cpu')  # You define this
        predictions.append(pred)
    return plot_boxes_on_image(img_path, boxes, predictions)





if __name__ == '__main__':
    image_path = r'images/4a7c7e0f-2225-4cd5-9570-8590e00a4b70.jpeg'
    model = YOLO(r"scripts/runs/detect/train15/weights/best.pt")
    output_img = test_by_crop(model, image_path, conf = 0.03, out_conf=0.5)
    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, output_img)
    # Open with default image viewer
    if platform.system() == "Windows":
        os.startfile(output_path)    




#list = [[3179, 1675, 3237, 1732], [568, 1477, 618, 1551], [2535, 3015, 2596, 3062], [2715, 894, 2760, 952], [2482, 2329, 2539, 2381], [2648, 1390, 2710, 1444], [1586, 2123, 1641, 2185], [2224, 2860, 2281, 2916], [2930, 3937, 2973, 3994], [2745, 1139, 2782, 1189], [2125, 2181, 2180, 2253], [1549, 2572, 1606, 2616], [3293, 933, 3336, 983], [2863, 3631, 2925, 3674], [1046, 3399, 1092, 3439], [2613, 1080, 2664, 1127], [2434, 1503, 2463, 1540], [2565, 4455, 2610, 4490], [850, 3442, 892, 3485], [2925, 2550, 2962, 2583], [1284, 4040, 1322, 4078], [2322, 2998, 2367, 3030], [1155, 3218, 1197, 3248], [430, 981, 458, 1013], [1656, 4936, 1699, 4968], [2904, 1901, 2946, 1937], [3896, 1210, 3926, 1246], [1201, 2613, 1238, 2642], [1253, 3238, 1290, 3260], [3655, 795, 3681, 821], [2580, 3948, 2617, 3975], [875, 3946, 899, 3981], [10, 1408, 39, 1443], [3034, 1869, 3082, 1913], [3111, 1481, 3135, 1507], [1499, 5040, 1531, 5068], [608, 611, 641, 632], [2039, 3481, 2079, 3516], [1113, 1429, 1139, 1458], [3538, 1289, 3565, 1315], [3977, 755, 4007, 780], [2414, 2233, 2439, 2258], [1260, 666, 1289, 705], [3104, 2721, 3310, 2913], [1923, 1657, 1948, 1683], [2662, 611, 2684, 639], [2930, 1536, 2951, 1563], [653, 2653, 682, 2696], [887, 3377, 912, 3405], [3496, 683, 3518, 707], [1722, 1812, 1747, 1839], [2657, 3382, 2695, 3405], [939, 4568, 969, 4600], [933, 4423, 962, 4461], [3241, 625, 3267, 648], [3030, 724, 3055, 755], [3324, 1160, 3355, 1189], [1507, 1590, 1527, 1613], [3341, 1615, 3362, 1638], [1389, 4760, 1417, 4782], [2355, 4843, 2387, 4860], [2959, 3502, 2998, 3538], [630, 1752, 650, 1771], [2561, 2882, 2588, 2900], [2867, 4073, 2912, 4113], [1855, 5088, 1880, 5106], [1815, 4372, 1842, 4390], [1068, 4300, 1107, 4329], [2974, 3624, 3001, 3647], [1472, 1158, 1496, 1187], [2068, 3877, 2093, 3902], [1048, 4310, 1081, 4334], [3523, 1577, 3541, 1598], [1327, 997, 1352, 1022], [2882, 252, 2914, 272], [1703, 175, 1735, 202], [2290, 1778, 2328, 1823], [3756, 1203, 3776, 1228], [2977, 522, 2994, 539], [3919, 732, 3940, 753], [2187, 3556, 2208, 3576], [914, 4505, 935, 4533]]

# def load_yolo_boxes(txt_path, img_width, img_height):
#     boxes = []
#     with open(txt_path, 'r') as f:
#         for line in f:
#             parts = list(map(float, line.strip().split()))
#             if len(parts) == 5:
#                 _, x_center, y_center, w, h = parts  # Ignore class
#                 x1 = int((x_center - w / 2) * img_width)
#                 y1 = int((y_center - h / 2) * img_height)
#                 x2 = int((x_center + w / 2) * img_width)
#                 y2 = int((y_center + h / 2) * img_height)
#                 boxes.append([x1, y1, x2, y2])
#     return boxes
# def draw_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
#     img = image.copy()
#     for i, box in enumerate(boxes):
#         c, x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
#         if labels and i < len(labels):
#             label = labels[i]
#             cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, color, 1, cv2.LINE_AA)
#     return img


# def load_boxes_from_txt(txt_path):
#     boxes = []
#     with open(txt_path, 'r') as f:
#         for line in f:
#             parts = list(map(int, line.strip().split()))
#             if len(parts) == 4:
#                 boxes.append(parts)
#     return boxes


# def evaluate(yolo_model):
#     # Evaluate the model's performance on the validation set
#     return yolo_model.val(data="../datasets/lesions.yaml")


# def test_img(yolo_model, img_path, size):
#     # Perform object detection on an image
#     results = yolo_model(img_path, imgsz = size, conf=0.01)  # Predict on an image
#     results_class = type(results[0])
#     # Print the source code of the 'show' method
#     img = results[0].orig_img
#     # Convert BGR (OpenCV) to RGB (matplotlib)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.show()
#    # results[0].show()


# def export_onnx(yolo_model):
#     # Export the model to ONNX format for deployment
#     return yolo_model.export(format="onnx")  # Returns the path to the exported model


# def plot_gt(image_path, label_path):
#     # Load image to get dimensions
#     image = cv2.imread(image_path)
#     height, width = image.shape[:2]

#     # Load boxes and plot
#     boxes = load_yolo_boxes(label_path, width, height)
#     output_img = plot_boxes_on_image(image_path, boxes)



#     output_path = "output_with_boxes.jpg"
#     cv2.imwrite(output_path, output_img)

#     # Open with default image viewer
#     if platform.system() == "Windows":
#         os.startfile(output_path)


# def loop_in_crops(input_folder, model, image_size):
#     for filename in os.listdir(input_folder):
#         path = os.path.join(input_folder, filename)
#         test_img(model, path, image_size)


# def detect_and_plot(yolo_model, image_path, size, conf=0.1):
#     # Load image to get dimensions
#     results = yolo_model(image_path, imgsz=size, conf=conf)
#     # Load boxes and plot
#     boxes = results[0].boxes.xyxy.squeeze().detach()
#     boxes_list = boxes.cpu().int().tolist()
#     output_img = plot_boxes_on_image(image_path, boxes_list)



#     output_path = "output_with_boxes.jpg"
#     cv2.imwrite(output_path, output_img)

#     # Open with default image viewer
#     if platform.system() == "Windows":
#         os.startfile(output_path)  
# 
# 
# def main(model, image_path, image_size, input_folder=None, label_path = None):
#     # Loop over the crops
#     if input_folder is not None:
#         loop_in_crops(input_folder, model, image_size)
#         return
#     # Plot the ground truth
#     if label_path is not None:
#         plot_gt(image_path, label_path)
#         return
#     # Test one image
#     test_img(model, image_path, image_size)    