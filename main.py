import PIL.Image
import torch
import os
import matplotlib
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from transformers import SamModel, SamProcessor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from sam_utils import (
    show_masks_on_image, 
    show_boxes_on_image, 
    resize_pil_image,
    show_points_on_image,
    find_center,
    load_image
    )


class OwlViTWrapper:
    def __init__(self, obj_detect_threshold:float=0.1):
        self.obj_detect_threshold = obj_detect_threshold
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.text: list[list[str]] = None
        
    def _process_result(self, results):
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = self.texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        boxes_post_process = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            boxes_post_process.append(box)
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            
        centers = []
        for box in boxes_post_process:
            center = find_center(box)
            centers.append(center)
            print(f"Center of the detected {text} is at {center}")
            
        return boxes_post_process, centers
    
            
    def detect_objects(self, image: PIL.Image, texts: list[list[str]] = None) -> dict:
        self.texts = texts
        target_sizes = torch.Tensor([image.size[::-1]])
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.obj_detect_threshold)
        boxes, centers = self._process_result(results)
        return boxes, centers
        
class SAMWrapper:
    def __init__(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
    def segment_objects(self, image: PIL.Image, input_points=None, input_boxes=None) -> dict[str, torch.Tensor]:
        
        inputs = self.processor(image, return_tensors="pt").to(device)
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        inputs = self.processor(image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt").to(device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks, scores
        
        
def process_image(image_path, owl_detector, sam_setmenter, output_dir):
    # Load the image
    raw_image = load_image(path=image_path,resizing_ratio=1)
    
    # Generate prompts
    # prompt = [["a photo of a tree"]]
    prompt = [["an single tree"]]
    
    # Object detection
    boxes, points = owl_detector.detect_objects(image=raw_image, texts=prompt)
    
    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output subdirectory for this image
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Save points visualization
    show_points_on_image(raw_image, points, save_on=os.path.join(image_output_dir, 'image(points).jpg'), title=prompt[0][0])
    
    # Save boxes visualization
    show_boxes_on_image(raw_image, boxes, save_on=os.path.join(image_output_dir, 'image(box).jpg'), title=prompt[0][0])
    
    # Segmentation with points
    input_points = [points]
    masks_p, scores_p = sam_setmenter.segment_objects(image=raw_image, input_points=input_points)
    show_masks_on_image(raw_image, masks_p[0], scores_p, segment_type="points", save_on=os.path.join(image_output_dir, 'image(seg-points).jpg'))
    
    # Segmentation with boxes
    input_boxes = [boxes]
    masks_b, scores_b = sam_setmenter.segment_objects(image=raw_image, input_boxes=input_boxes)
    show_masks_on_image(raw_image, masks_b[0][0], scores_b[:, 0, :], segment_type="boxes", save_on=os.path.join(image_output_dir, 'image(seg-boxes).jpg'))

if __name__ == "__main__":
    matplotlib.use('Agg')  # Avoids opening a GUI for plotting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    
    # Initialize detectors
    owl_detector = OwlViTWrapper()
    sam_setmenter = SAMWrapper()
    
    # Define input and output directories
    input_dir = r'C:\Users\Maods\Documents\Development\Mestrado\project-tree\data\segmentation_test'
    output_dir = './data/output_seg'
    
    # Loop over all images in the input directory
    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, image_file)
            try:
                process_image(image_path, owl_detector, sam_setmenter, output_dir)
            except:
                continue
        

    
    