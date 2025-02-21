from IPython.display import display
from PIL import Image
import os
import base64
import numpy as np
import cv2,torch
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Check if CUDA is available
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print(f"torch cuda available: {torch.cuda.is_available()}")

# path = os.getenv('wflowhome')
# pythonpath = os.path.join(path, '.python')
processor = TrOCRProcessor.from_pretrained(os.path.join('ocrmodel','processor'))
## "microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained('ocrmodel')
yolo_model = YOLO('best.pt')

def roi(batch_im0):
    names = yolo_model.names
    # Perform object detection on batch of images
    if len(batch_im0) == 0:
        return [],[]
    batch_results = yolo_model.predict(batch_im0, show=False)
    #print(batch_results)
    all_crop_obj = []  # Store cropped objects from all images
    Empty_image = [] # Store the images which have no detection

    for im0, results in zip(batch_im0, batch_results):
        crop_obj = []
        if not results:
            Empty_image.append(results.path) # Store the empty image name
            continue  # Skip to next image if no objects detected

        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        annotator = Annotator(im0, line_width=2, example=names)
        classes_to_extract = [0]

        for box, cls in zip(boxes, clss):
            if int(cls) in classes_to_extract:
                annotator.box_label(box, color=colors(
                    int(cls), True), label=names[int(cls)])
                crop_img = im0[int(box[1]):int(box[3]),
                                int(box[0]):int(box[2])]
                crop_obj = Image.fromarray(crop_img, mode='RGB')

        all_crop_obj.append(crop_obj)  # Store cropped objects for this image
    return all_crop_obj,Empty_image

def predict(batch_images):
    # List to store generated texts for each image
    generated_texts = []
    for eachImage in batch_images:
        # Process each image to get pixel_values
        try:
            pixel_values = processor(eachImage, return_tensors="pt")
            # Generate text using the model
            generated_ids = model.generate(**pixel_values)
        except IndexError :
            generated_ids = None
        if generated_ids is None:
            generated_texts.append(None)
        else:
            # Decode generated text
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True)[0]
            generated_texts.append(generated_text)
    return generated_texts

def predictBatch(batch_images,Empty_image):
    # display(image)
    if len(batch_images) == 0 : 
        nodection = []
        for i in Empty_image:
            filename = i
            number = int(filename.split('image')[1].split('.')[0])
            nodection.insert(number,"No detection")
        return nodection
    try :
        pixel_values = processor(batch_images, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
    except ValueError as ex:
        print (ex)
        return predict(batch_images)
    if generated_ids is None:
        return None
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)
    for i in Empty_image:
        filename = i
        number = int(filename.split('image')[1].split('.')[0])
        generated_text.insert(number,"No detection")

    return generated_text
    
def decodeBitmaptoimage(data):
    decoded_data = base64.b64decode(data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    return image


def readImageFromFile(image_path):
    im0 = cv2.imread(image_path)
    return im0
