import numpy as np
import streamlit as st
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

device = torch.device('cpu')
PATH = "model.pt"
class_names = ['COVID-19', 'NORMAL']

def CNN_Model(pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    return model

model = CNN_Model(pretrained=True)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()


def predict(image):
	with torch.no_grad():
		mean_nums = [0.485, 0.456, 0.406]
		std_nums = [0.229, 0.224, 0.225]

		img_transforms = transforms.Compose([
                                transforms.Resize((150,150)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_nums, std=std_nums)
		])
	
		image = image.convert('RGB')
		image = img_transforms(image)
		pred = model(image.unsqueeze(dim=0))
		proba = F.softmax(pred, dim=1)
		pred = np.argmax(proba, axis=1)

		return pred.item(), proba



def main():
	st.title("COVID-19 detection from X-ray images using Deep Learning")

	uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
	
	if uploaded_file is not None:
	    image = Image.open(uploaded_file)

	    st.image(image.resize((299, 299)), caption='Uploaded Image.')
	    st.write("")
	    st.write("Classifying...")
	    
	    pred, proba = predict(image)

	    st.write("Probability of COVID-19: " + str(round(proba[0][0].item()*100, 2)) + "%")
	    st.write("Probability of NORMAL: " + str(round(proba[0][1].item()*100, 2)) + "%")

	    if pred == 0:
	    	st.error(class_names[0])
	    else:
	    	st.success(class_names[1])



if __name__ == '__main__':
	main()