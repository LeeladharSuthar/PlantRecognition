from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

dynamodb = boto3.client('dynamodb', 
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name='us-east-1',
                        verify=False)

table_name = 'Plants'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('./plant_classifier_10.pth', map_location=device)

model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Rebuild model architecture
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(checkpoint['class_names']))
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

class_names = checkpoint['class_names']
print("Model and class names loaded!")

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

def predict_image(image_path, model, class_names):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    transform = data_transforms['val']
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        return predicted_class, confidence.item()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper to validate file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# POST route to upload image
@app.route('/upload', methods=['POST'])
def upload_file():
    print("Request received")
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the file before prediction

        # Make the prediction
        predicted_class, confidence = predict_image(file_path, model, class_names)
        
        part1, part2 = predicted_class.split('_', 1)
        
        # Initialize default values for uses and image URLs
        uses = []
        image_urls = []

        # Fetch data from DynamoDB only if part1 is 'medicinal'
        if part1 == 'medicinal':
            try:
                response = dynamodb.get_item(
                    TableName=table_name,
                    Key={'name': {'S': part2}}
                )

                if 'Item' in response:
                    # Extract the 'uses' and 'imageUrl' data from the response
                    uses = [item['S'] for item in response['Item']['uses']['L']]
                    image_urls = [item['S'] for item in response['Item']['imageUrl']['L']]
                    print("Uses:", uses)
                    print("Image URLs:", image_urls)
                else:
                    print(f"No data found for {part2}")
            except (NoCredentialsError, PartialCredentialsError) as e:
                return jsonify({'error': 'AWS credentials error'}), 500
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # Delete the uploaded image after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted image: {file_path}")

        # Return response with medicinal data if available
        if predicted_class:
            return jsonify({
                'name': part2,
                'confidence': confidence,
                'type': 'Medicinal' if part1 == 'medicinal' else 'Non Medicinal',
                'uses': uses,
                'imageUrl': image_urls
            }), 200
        else:
            return jsonify({'error': 'Prediction failed'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)