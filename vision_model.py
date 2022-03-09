import torch.nn as nn
import os
import torch
from efficientnet_pytorch import EfficientNet

class Vision_Model(nn.Module):
    def __init__(self, n_classes=5,n_latient=768, pretrained=False, dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"):

        super(Vision_Model, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_latient)
        self.classifier = nn.Linear(n_latient, n_classes)

        pretrained = True
        if pretrained:
            model_path = os.path.join(dir_base, 'Zach_Analysis/models/teacher_student/pretrained_student_vision_model')
            self.model.load_state_dict(torch.load(model_path))
            print("is using the wieghts stored at this location")
        else:
            print("doesn't use saved weights, using random weights in vision")
        #self.model.head = nn.Linear(self.model.head.in_features, n_classes)

        #self.model.head = nn.Linear(self.model.head.in_features, 512)

    def forward(self, x):
        x = self.model(x)
        output = self.classifier(x)
        return output