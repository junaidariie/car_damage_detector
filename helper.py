import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

CLASSES = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
NUM_CLASSES = len(CLASSES)



# 2. TRANSFORMS

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



# 3. MODEL CLASS

class CarClassifierResnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)



# 4. LOAD MODEL ONLY ONCE

def load_model():
    model = CarClassifierResnet(NUM_CLASSES)
    model.load_state_dict(torch.load("Model_Resnet.pth", map_location="cpu"))
    model.eval()
    return model

MODEL = load_model()
TRANSFORM = get_inference_transform()



# 5. PREDICTION FUNCTION

def predict_image(pil_img):
    img_tensor = TRANSFORM(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = MODEL(img_tensor)
        _, predicted_idx = torch.max(outputs, dim=1)
        return CLASSES[predicted_idx.item()]

