import torch
import torch.nn.functional as F
import torchvision.models as models
import transformers
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

class ImageModelResNet50(torch.nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.resnet50 = models.resnet50(pretrained=True)
        self.avgpool = self.resnet50.avgpool
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim))

    def forward(self, x):
        out_7x7 = self.resnet50(x).view(-1, 2048, 7, 7)
        out = self.avgpool(out_7x7).view(-1, 2048)
        # critical to normalize projections
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        # return out, attn, residual
        return out


class ImageModelResNet101(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.resnet101 = models.resnet101(pretrained=True)
        self.avgpool = self.resnet101.avgpool
        self.resnet101 = torch.nn.Sequential(*list(self.resnet101.children())[:-2])
        for param in self.resnet101.parameters():
            param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim))

    def forward(self, x):
        out_7x7 = self.resnet101(x).view(-1, 2048, 7, 7)
        out = self.avgpool(out_7x7).view(-1, 2048)
        # critical to normalize projections
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        # return out, attn, residual
        return out


class ImageModelVGG16(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.vgg = models.vgg16(pretrained=True)
        self.avgpool = self.vgg.avgpool
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:-1])
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, 1024, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim, bias=True))

        # self.vgg16.classifier = self.fc

    def forward(self, x):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=-1)
        x = self.fc(x)
        return x



class TextModel(torch.nn.Module):
    def __init__(self, out_dim, freeze_bert=True):
        super(TextModel, self).__init__()
        self.out_dim = out_dim
        self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(1024, out_dim))

    def forward(self, input_ids, attention_mask):

        _, out = self.bert_model(input_ids= input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.fc(out)
        return out


class MultiModelResnet50(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_bert=True):
        super(MultiModelResnet50, self).__init__()

        self.out_dim = out_dim
        self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.resnet50 = models.resnet50(pretrained=True)
        self.avgpool = self.resnet50.avgpool
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.batchnorm = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

    def forward(self, input_ids, attention_mask, image):
        _, text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        out_7x7 = self.resnet50(image).view(-1, 2048, 7, 7)
        image_outputs = self.avgpool(out_7x7).view(-1, 2048)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out

class MultiModelResnet101(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_bert=True):
        super(MultiModelResnet101, self).__init__()

        self.out_dim = out_dim
        self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.resnet101 = models.resnet101(pretrained=True)
        self.avgpool = self.resnet101.avgpool
        self.resnet101 = torch.nn.Sequential(*list(self.resnet101.children())[:-2])
        for param in self.resnet101.parameters():
            param.requires_grad = False

        self.batchnorm = torch.nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

    def forward(self, input_ids, attention_mask, image):
        _, text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        out_7x7 = self.resnet101(image).view(-1, 2048, 7, 7)
        image_outputs = self.avgpool(out_7x7).view(-1, 2048)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out

class MultiModelVGG16(torch.nn.Module):
    def __init__(self, out_dim=1, freeze_bert=True):
        super(MultiModelVGG16, self).__init__()

        self.out_dim = out_dim
        self.bert_model = transformers.BertModel.from_pretrained("bert-large-uncased")
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.vgg = models.vgg16(pretrained=True)
        self.avgpool = self.vgg.avgpool
        self.vgg = torch.nn.Sequential(*list(self.vgg.children())[:-1])
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, 1024, bias=True))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(256, out_dim))

    def forward(self, input_ids, attention_mask, image):
        _, text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        image_outputs = self.vgg(image)
        image_outputs = torch.flatten(image_outputs, 1)
        image_outputs = F.normalize(image_outputs, dim=-1)
        image_outputs = self.projector(image_outputs)

        multi_outputs = torch.cat((text_outputs, image_outputs), 1)
        out = self.fc(multi_outputs)

        return out

