import torch
import clip
from torchvision import models, transforms
from torchvision.models import ResNet101_Weights
import numpy as np
from PIL import Image

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

def extract_embedding(image: Image.Image, description: str = None) -> np.ndarray:
    # Process image
    clip_img = preprocess_clip(image).unsqueeze(0).to(device)
    resnet_img = resnet_transform(image).unsqueeze(0)

    with torch.no_grad():
        # Image embeddings
        clip_emb = clip_model.encode_image(clip_img).cpu().numpy().flatten()
        resnet_emb = resnet_model(resnet_img).squeeze().numpy().flatten()

        # Normalize
        clip_emb = clip_emb / np.linalg.norm(clip_emb)
        resnet_emb = resnet_emb / np.linalg.norm(resnet_emb)

        combined_emb = np.concatenate([clip_emb, resnet_emb])

        # Optional: add text embedding if description is provided
        if description:
            text_tokens = clip.tokenize([description]).to(device)
            text_emb = clip_model.encode_text(text_tokens).cpu().numpy().flatten()
            text_emb = text_emb / np.linalg.norm(text_emb)
            combined_emb = np.concatenate([combined_emb, text_emb])
        else:
            text_emb = np.zeros(512)

    return combined_emb.astype("float32")
