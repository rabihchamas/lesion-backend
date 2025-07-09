import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch
import torch.nn as nn 


class Backbone(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.base_model.classifier = nn.Linear(in_features=self.base_model.classifier.in_features,
                                               out_features=num_classes, bias=True)


    def forward(self, x):
        x = self.base_model(x)
        return x

    def show_count_para(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters of the model: {n_parameters / 1e6:.2f} M")
        return n_parameters


transform_val = A.Compose([
    A.Resize(224, 224),  # image_size must be defined
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def get_classifier(device, checkpoint_path):
    model = Backbone()
    # model.eval()
    model = model.to(device=device)
    # checkpoint_path = r'checkpoints/efficientnet_b0_8e-0.156_2025-06-30.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def cnn_model_predict(model, crop, device):
    # model = get_model(device=device)
    # Apply Albumentations transform
    transformed = transform_val(image=crop)
    crop_tensor = transformed['image'].unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        output = model(crop_tensor)
        mel_pred = output[:,1]
        prob = torch.sigmoid(mel_pred).item()  # For binary classification
    return prob


if __name__ == "__main__":
    device=torch.device("cpu")
    model = get_model(device)
