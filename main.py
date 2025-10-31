import cv2
import time
from core.vision import VisionBrain
from core.speech import SpeechEngine

def main():

    vision_ai = VisionBrain()
    speech_engine = SpeechEngine()
    

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("VisionBrain Started!")
    print("Press 'q' to quit")
    print("Press 's' to force scene description")
    print("Press 'space' to toggle auto-description")
    
    last_analysis_time = 0
    analysis_interval = 15 
    auto_describe = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
     
        if auto_describe and (current_time - last_analysis_time) > analysis_interval:
            description, objects = vision_ai.understand_scene(frame)
            print(f"\nScene Analysis: {description}")
            speech_engine.text_to_speech(description)
            last_analysis_time = current_time
        
      
        _, objects = vision_ai.understand_scene(frame)
        frame_with_boxes = vision_ai.draw_detections(frame, objects)
        
     
        status = "AUTO-DESCRIBE: ON" if auto_describe else "AUTO-DESCRIBE: OFF"
        cv2.putText(frame_with_boxes, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_with_boxes, "Press 'q' to quit, 's' to describe, 'space' to toggle auto", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        
        cv2.imshow('VisionBrain - Real-time AI Vision By Wasif', frame_with_boxes)
        
     
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            
            description, objects = vision_ai.understand_scene(frame)
            print(f"\nManual Scene Analysis: {description}")
            speech_engine.text_to_speech(description)
            last_analysis_time = current_time
        elif key == ord(' '):
           
            auto_describe = not auto_describe
            print(f"Auto-description: {'ON' if auto_describe else 'OFF'}")
    
 
    cap.release()
    cv2.destroyAllWindows()
    speech_engine.stop_speech()
    print("VisionBrain stopped.")

if __name__ == "__main__":
    main()