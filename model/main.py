import torch
from torch import nn
from torchvision import datasets, models, transforms
from .classes import model_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def __model_architect():
    model_TL = models.densenet161(pretrained=True)

    # Freeze All layers
    for param in model_TL.parameters():
        param.requires_grad = False
    for param in model_TL.classifier.parameters():
        param.requires_grad = True
        
    number_features = model_TL.classifier.in_features   # for desnet pretrained model
    model_TL.classifier = torch.nn.Linear(in_features=number_features, out_features=196)

    return model_TL



def __load_model():

    model = __model_architect().to(device)
    
    checkpoint = torch.load(r"C:\Users\saadr\OneDrive\Desktop\files\MLWebApps\CarClassifier_webapp\model.pth", map_location=device)


    model.load_state_dict(checkpoint['model_state_dict'])
    

    return model


def __class_dict(classes):
    dict_ind_classes = {i: classes[i]  for i in range(len(classes))} 
    return dict_ind_classes



def predict_car_model(img):
# it seems not normalized, so we need to normalize the dataset 
    mean = [0.4708, 0.4602, 0.4549]
    std = [0.2891, 0.2882, 0.2967]
    transform = transforms.Compose([
    
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dict_classes = __class_dict(model_classes)
    model = __load_model().to(device)

    transform_img = transform(img)
    transform_img = torch.unsqueeze(transform_img, dim=0).to(device)


    model.eval()
    with torch.no_grad():
        scores = model(transform_img)
        probs = torch.nn.functional.softmax(scores, dim=1)

        predicted_index = torch.argmax(probs, dim=1)
    

    print("predicted_index: ",predicted_index)
    return dict_classes[int(predicted_index)]

