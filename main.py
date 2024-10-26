import cv2
from ultralytics import YOLO
import google.generativeai as genai
import os 
from dotenv import load_dotenv
from IPython.display import Markdown,display
import textwrap

model = YOLO('yolov8n.pt')  

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)



def to_markdown(text):
  text = text.replace("*", " ")
  text=text.strip("\n")
  return Markdown(textwrap.indent(text, ">  ",predicate=lambda _:True))

config={
    "temperature": 0.7,  
    "max_tokens": 100,  
    "top_p": 0.9, 
    "top_k": 40  
}

llm = genai.GenerativeModel(model_name="gemini-pro",generation_config=config)


cap = cv2.VideoCapture(0)  


detected_labels= []
while True:
    ret, frame = cap.read()  
    if not ret:
        break

    
    results = model(frame)

    
    for result in results:
        boxes = result.boxes  
        for box in boxes:
           
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            
            cls = int(box.cls[0])  
            conf = box.conf[0]  
            class_name = model.names[cls]  

            if class_name not in detected_labels:
                detected_labels.append(class_name)

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("YOLOv8 Object Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


labels_for_story = ", ".join(detected_labels)

prompt = f"You are an expert story writer,please provide story about {labels_for_story}.Make it more dramatic.Also you may suggest a song.Also make it turkish please."

response=llm.generate_content(prompt)
print(to_markdown(response.text))