import torch
import clip
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet101_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/16", device=device)

resnet_model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_model.eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_embedding(image: Image.Image) -> np.ndarray:
    clip_img = preprocess_clip(image).unsqueeze(0).to(device)
    resnet_img = resnet_transform(image).unsqueeze(0)

    with torch.no_grad():
        clip_emb = clip_model.encode_image(clip_img).cpu().numpy().flatten()
        resnet_emb = resnet_model(resnet_img).squeeze().numpy().flatten()

    clip_emb = clip_emb / np.linalg.norm(clip_emb)
    resnet_emb = resnet_emb / np.linalg.norm(resnet_emb)
    combined_emb = np.concatenate([clip_emb, resnet_emb]).astype("float32")

    return combined_emb
