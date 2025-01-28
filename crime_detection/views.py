from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor once (replace with your model path)
model = ViTForImageClassification.from_pretrained("path_to_trained_model")
processor = ViTImageProcessor.from_pretrained("path_to_trained_model")

class PredictView(APIView):
    def post(self, request):
        file = request.FILES['image']  # Expecting an uploaded image
        image = Image.open(file)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_label = model.config.id2label[predictions.argmax().item()]
            confidence = predictions.max().item()

        # Return prediction response
        return Response({"label": predicted_label, "confidence": confidence})

