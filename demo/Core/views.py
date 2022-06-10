from . import models
from django.shortcuts import render

# Import DataLoader and corresponding libraries
import torchvision.transforms as TT
from PIL import Image

# For model construction
from collections import OrderedDict

# Import libraries for tensors
import numpy as np
import torch
import torch.nn as nn
import os

# Prototype of model 2.
# ResNet50 outputs (Batchsize, 1000) tensor as output, so we reduce them to 397.
class Model2(nn.Module):
    def __init__(self, num_classes=397):
        super().__init__()
        self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1000, num_classes, bias=True)
    
    def forward (self, x):
        x = self.resnet50(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

# Transform for image
transform = TT.Compose([
        TT.Resize([128, 281]),
        TT.ToTensor(),
        TT.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        uploadedFile = request.FILES["uploadedFile"]

        nocall_model = nn.Sequential(OrderedDict([
            ("resnet50", torch.hub.load(
                'pytorch/vision:v0.10.0', 'resnet50', pretrained=True)),
            ("relu", nn.ReLU()),
            ("linear", nn.Linear(1000, 2, bias=True)),
            ("softmax", nn.Softmax(dim=-1))
        ]))
        classifier_model = Model2()

        # Load checkpoint
        if os.path.exists('./nocall_detector.pt'):
            ckpt = torch.load('./nocall_detector.pt')
            nocall_model.load_state_dict(ckpt)
        
        if os.path.exists('./bird_species_classifier.pt'):
            ckpt = torch.load('./bird_species_classifier.pt')
            classifier_model.load_state_dict(ckpt['model_state_dict'])

        # Load file
        source = np.load('./media/Uploaded Files/' + fileTitle)

        # Rearrange numpy arrays
        source = source.transpose(1, 2, 0)
        
        # Add RGB dimension & augmentation
        source = np.stack((np.squeeze(source), ) * 3, -1)
        source = transform(Image.fromarray(source))

        nocallWeight = nocall_model(source)[0][1]
        inferenceIndex = torch.argmax(classifier_model(source), dim=-1)

        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            uploadedFile = uploadedFile,
            nocallWeight = nocallWeight,
            inferenceIndex = inferenceIndex
        )
        document.save()

    documents = models.Document.objects.all()

    return render(request, "Core/upload-file.html", context = {
        "files": documents
    })