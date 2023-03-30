# ArgMining2022-ImageArg

Repo for our paper: [ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining](https://aclanthology.org/2022.argmining-1.pdf#page=13)

## Run experiments
Please download tweet content using [TwitterAPI](https://developer.twitter.com/en/docs/twitter-api) into the root directory `data` before run the experiments. The following parameters need to config to run specific experiments. Please check the code for details.

```angular2html
--data-dir', default='./data', help='path to data'
--exp-dir', default='./experiments/debug', help='path save experimental results'
--exp-mode', default=5, choices=[0,1,2,3,4,5,6], type=int, help='0:persuasive; 1:image content; 2: stance; 3: perusasion_mode; 4:persuasive mode logos; 5:persuasive mode panthos; 6:persuasive mode ethos'
--num-epochs', default=10, type=int, help='number of running epochs'
--data-mode', default=2, choices=[0,1,2], type=int, help='0:text; 1:image; 2:image+text'
--gpus', default='0', type=str, help='specified gpus'
--seed', default=22, type=int, help='random seed number'
--batch-size', default=16, type=int, help='number of samples per batch'
--lr', default=0.001, type=float, help='learning rate'
--persuasive-label-threshold', default=0.6, type=float, help='threshold to categorize persuasive labels'
--kfold', default=5, help='number of fold validation'
--img-model', default=0, choices=[0,1,2], type=int, help='0:Resnet50; 1:Resnet101; 2:VGG16'
--save-checkpoint', default=0, choices=[0,1], type=int, help='0:do not save checkpoints; 1:save checkpoints'
--skip-non-persuasion-mode', default=0, choices=[0,1], type=int, help='0:do not skip; 1:skip'
```

* Run image modality experiments
```angular2html
python main_image.py \
  --num-epochs=3  \
  --batch-size=16  \
  --exp-mode=0 \
  --data-mode=1 \
  --lr=0.001  \
  --img-model=0 \
  --persuasive-label-threshold=0.5 \
  --save-checkpoint=0 \
  --skip-non-persuasion-mode=1
```
* Run text modality experiments
```angular2html
python main_text.py \
  --num-epochs=3  \
  --batch-size=16  \
  --exp-mode=0 \
  --data-mode=0 \
  --lr=0.001  \
  --img-model=0 \
  --persuasive-label-threshold=0.5 \
  --save-checkpoint=0 \
  --skip-non-persuasion-mode=1
```
* Run multimodality experiments
```angular2html
python main_multimodality.py \
  --num-epochs=3  \
  --batch-size=16  \
  --exp-mode=0 \
  --data-mode=2 \
  --lr=0.001  \
  --img-model=0 \
  --persuasive-label-threshold=0.5 \
  --save-checkpoint=0 \
  --skip-non-persuasion-mode=1
```

