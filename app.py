from flask import Flask, request,jsonify,json
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from paddleocr import PaddleOCR
from PIL import Image
import shutil
import warnings

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'temp_file'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def clean_decimal_string(input_string):
    # Remove all spaces from the input string
    input_string = input_string.replace(" ", "")
    
    # Keep only the digits from the input string
    digits_only = ''.join(filter(str.isdigit, input_string))
    
    # If there are fewer than 3 digits, the result will just be the string of digits
    if len(digits_only) < 3:
        return digits_only
    
    # Place a decimal point two places from the end of the string
    result = digits_only[:-2] + '.' + digits_only[-2:]
    
    return result

# Float check
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def test_single_image(model, image_path):
    # Define transformation to apply to the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor()            # Convert to tensor
    ])

    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Use the model to make predictions
    model.eval()
    with torch.no_grad():
        # Move the preprocessed image to the appropriate device (CPU or GPU)
        device = next(model.parameters()).device
        image = image.to(device)

        # Perform forward pass to get the model output
        output = model(image)

        # Apply sigmoid activation to convert output to probability
        probability = torch.sigmoid(output).item()
        return probability

content_path='xxx'

@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method=='POST':
        try:
            # my_set = set()
            my_list=[]
            file=request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Detecting and Saving Cropped Images.
            temp_image_path=os.path.join('temp_file', filename)
            model.predict(temp_image_path,save_crop=True,project="xxx", name="yyy") 

            # Set the directory containing your images
            
            image_dir = "xxx/yyy/crops/Line_1"
            display_dir="xxx/yyy/crops/Display"


            display_files = [os.path.join(display_dir, file) for file in os.listdir(display_dir) if file.endswith(('jpg', 'jpeg' 'png'))]

            # Check if there is exactly one display
            if len(display_files) != 1:
                return jsonify("x")

            # Get a list of all image files in the directory
            image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg' 'png'))]

            # ANOMALY DETECTION STARTS HERE

            test_image_path = display_files[0]
            # Test the model on a single image
            probability=test_single_image(anomaly_model, test_image_path)
            if probability > 0.5:
                return jsonify("x1")
            # ANOMALY DETECTION ENDS HERE

            # PaddleOCR from Here
            for image_path in image_files:
                # Perform OCR
                result = ocr.ocr(image_path, det=False, rec=True)
                if result:
                    for res in result:
                        for line in res:
                            numbers=clean_decimal_string(line[0])
                            if isfloat(numbers):
                                my_list.append(float(clean_decimal_string(line[0])))      
                else:
                    shutil.rmtree(content_path)
                    jsonify({"error": "File upload failed or file type not allowed."}), 400


            my_list_sorted=sorted(my_list)
            if(len(my_list)!=2):
                return jsonify("x")
            final_list=[]
            # This block of code is for a single display fuel dispenser:
            new_obj={
                "Total Sale": my_list_sorted[1],
                "Price per Litre":my_list_sorted[0]
            }
            final_list.append(new_obj)
            return jsonify(final_list) 
            # This block of code is for a single display fuel dispenser
        except Exception as e:
            return jsonify("x")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(content_path):
                shutil.rmtree(content_path)


if __name__=='__main__':
    # Our trained Yolo model
    model = YOLO(r'best_w_display.pt')
    model_path = 'trained_model.pth'
    # Load the trained model
    anomaly_model = torchvision.models.resnet18(pretrained=False)
    device = torch.device('cpu')
    anomaly_model.to(device)
    num_ftrs = anomaly_model.fc.in_features
    anomaly_model.fc = nn.Linear(num_ftrs, 1)  # Change output layer for binary classification
    anomaly_model.load_state_dict(torch.load(model_path, map_location=device))
    # Initialize PaddleOCR
    ocr = PaddleOCR(rec_model_dir=r'Inference_Model', lang='en',show_log = False)  # Use once to download and load model into memory
    app.run(host='0.0.0.0', port=5000, debug=True)