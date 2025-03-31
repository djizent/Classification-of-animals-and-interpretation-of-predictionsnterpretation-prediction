import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from torchvision import models


def load_model():
    MODEL_PATH = "models/efficientnet_b0_animals.pth"
    device = torch.device("cpu")
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 90)
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
    model.eval()
    
    return model, device

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image_tensor, model, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_probs, top5_idx = torch.topk(probabilities, 5)
    return top5_probs.cpu().numpy(), top5_idx.cpu().numpy()

def predict_fn(images):
    model, device = load_model()
    images = (images * 255).astype(np.uint8)
    batch = torch.stack([transform_image(Image.fromarray(img)) for img in images])
    if batch.dim() == 5:  
        batch = batch.squeeze(1)
    batch = batch.to(device)
    with torch.no_grad():
        preds = model(batch)
        probs = torch.nn.functional.softmax(preds, dim=1)
    return probs.cpu().numpy()

def load_class_names():
    with open("class_names.txt", "r", encoding="utf-8") as file:
        class_names = file.readlines()
    return [line.strip() for line in class_names]

def main():
    st.title("Классификация животных")
    
    model, device = load_model()
    class_names = load_class_names()
    
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        image_tensor = transform_image(image).to(device)
        
        top5_probs, top5_idx = predict(image_tensor, model, device)

        st.subheader("Топ-5 предсказаний:")
        for i in range(5):
            predicted_label = class_names[top5_idx[0][i]]
            prob = top5_probs[0][i] * 100
            st.write(f"{i+1}. {predicted_label} - {prob:.2f}%")
        
        st.subheader("Объяснение LIME:")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(image).astype(np.float32) / 255.0, 
            predict_fn, 
            top_labels=5, 
            hide_color=0,
            num_samples=100
        )
        
        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0], 
            positive_only=True, 
            num_features=10,
            hide_rest=False
        )
        
        fig, ax = plt.subplots()
        ax.imshow(mark_boundaries(temp, mask, color=(1, 0, 0)))
        ax.axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()