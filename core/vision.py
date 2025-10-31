import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

class VisionBrain:
    def __init__(self):
        print("Loading Vision AI models...")
        
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        print("Models loaded successfully!")
    
    def understand_scene(self, frame):
        """Analyze frame and generate description"""
       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        results = self.yolo_model(pil_image)
        objects = results.pandas().xyxy[0]
        
        confident_objects = objects[objects['confidence'] > 0.5]
        object_list = confident_objects['name'].unique().tolist()

        if object_list:
            text_prompt = f"this image contains {', '.join(object_list[:3])}. Describe the scene in detail:"
        else:
            text_prompt = "Describe this image in detail:"
        
        inputs = self.processor(pil_image, text_prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=50, num_beams=5)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption, confident_objects
    
    def draw_detections(self, frame, objects):
        """Draw bounding boxes and labels on frame"""
        for _, obj in objects.iterrows():
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            label = f"{obj['name']} {obj['confidence']:.2f}"
            
         
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame