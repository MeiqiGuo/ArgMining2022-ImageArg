from PIL import Image
import torch
from utils import *
from torch.utils.data import Dataset
import preprocessor
import nltk
import json
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION, preprocessor.OPT.HASHTAG)
import warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    def __init__(self, args, annotation, root_dir, transform=None):
        self.annotation = annotation
        self.root_dir = root_dir
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_id = str(self.annotation.loc[idx, 'tweet_id'])
        img_path = os.path.join(self.root_dir,  f'{img_id}.jpg')
        try:
            # corrupted image
            image = Image.open(img_path).convert('RGB')
        except:
            return None
        if self.transform:
            image = self.transform(image)

        if self.args.exp_mode == 0:
            label = self.annotation.loc[idx, 'persuasiveness']
            label = encode_persuasiveness(label, self.args)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 1:
            label = self.annotation.loc[idx, 'content']
            label = encode_image_content_type(label)
            # skip "other" label
            if label == 6: return None
        elif self.args.exp_mode == 2: # stance
            label = self.annotation.loc[idx, 'stance']
            label = encode_stance(label)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 3: #
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot
            label = torch.FloatTensor(label)
        elif self.args.exp_mode == 4: # logos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[0]
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 5: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[1]
            label = torch.FloatTensor([label])
        else: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[2]
            label = torch.FloatTensor([label])

        return img_id, image, label


class TextDataset(Dataset):
    def __init__(self, args, annotation, root_dir, transform=None):
        self.annotation = annotation
        self.transform = transform
        annotation["tweet_text"] = annotation["tweet_text"].apply(lambda x: preprocessor.clean(x))
        self.input_ids, self.attention_masks = bert_tokenizer(annotation["tweet_text"].tolist())
        self.args = args
        self.root_dir = root_dir

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        text_id = str(self.annotation.loc[idx, 'tweet_id'])
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        img_id = str(self.annotation.loc[idx, 'tweet_id'])
        img_path = os.path.join(self.root_dir,  f'{img_id}.jpg')
        try:
            # corrupted image - even if for text only dataloader
            image = Image.open(img_path).convert('RGB')
        except:
            return None

        if self.args.exp_mode == 0:
            label = self.annotation.loc[idx, 'persuasiveness']
            label = encode_persuasiveness(label, self.args)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 1:
            label = self.annotation.loc[idx, 'content']
            label = encode_image_content_type(label)
            # skip "other" label
            if label == 6: return None
        elif self.args.exp_mode == 2: # stance
            label = self.annotation.loc[idx, 'stance']
            label = encode_stance(label)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 3: #
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot
            label = torch.FloatTensor(label)
        elif self.args.exp_mode == 4: # logos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[0]
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 5: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[1]
            label = torch.FloatTensor([label])
        else: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[2]
            label = torch.FloatTensor([label])

        return text_id, input_id, attention_mask, label


class ImageTextDataset(Dataset):
    def __init__(self, args, annotation, root_dir, transform=None):
        self.annotation = annotation
        annotation["tweet_text"] = annotation["tweet_text"].apply(lambda x: preprocessor.clean(x))
        self.input_ids, self.attention_masks = bert_tokenizer(annotation["tweet_text"].tolist())

        self.root_dir = root_dir
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        text_id = str(self.annotation.loc[idx, 'tweet_id'])
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        img_id = str(self.annotation.loc[idx, 'tweet_id'])
        img_path = os.path.join(self.root_dir,  f'{img_id}.jpg')
        try:
            # corrupted image
            image = Image.open(img_path).convert('RGB')
        except:
            return None
        if self.transform:
            image = self.transform(image)

        if self.args.exp_mode == 0:
            label = self.annotation.loc[idx, 'persuasiveness']
            label = encode_persuasiveness(label, self.args)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 1:
            label = self.annotation.loc[idx, 'content']
            label = encode_image_content_type(label)
            # skip "other" label
            if self.args.skip_non_persuasion_mode == 1:
                if label == 6: return None
        elif self.args.exp_mode == 2: # stance
            label = self.annotation.loc[idx, 'stance']
            label = encode_stance(label)
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 3: #
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot
            label = torch.FloatTensor(label)
        elif self.args.exp_mode == 4: # logos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[0]
            label = torch.FloatTensor([label])
        elif self.args.exp_mode == 5: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[1]
            label = torch.FloatTensor([label])
        else: # ethos
            label = self.annotation.loc[idx, 'persuasion_mode']
            label_onehot = encode_persuasive_mode(label)
            # skip no persuasion mode example
            if self.args.skip_non_persuasion_mode == 1:
                if sum(label_onehot) == 0: return None
            label = label_onehot[2]
            label = torch.FloatTensor([label])


        return text_id, input_id, attention_mask, image, label