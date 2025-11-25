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

class CarClassifierMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.features[-1].parameters():
            param.requires_grad = True


        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)




# 4. LOAD MODEL ONLY ONCE

def load_model():
    model = CarClassifierMobileNetV2(NUM_CLASSES)
    model.load_state_dict(torch.load("CarClassifierMobileNetV2", map_location="cpu"))
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
