# DNAF: Diffusion with Noise-Aware Feature (ICME2024)

## Results
<img src="assets/main_res.png">
From left to right are: pose input, source input, ground truth results, ADGAN results, PISE results, GFLA results, DPTN results, CASD results, NTED results, PIDM results, Our results.

<img src="assets/compare.jpg">
The following observations can be made that our model is able to: 

1) adeptly handle the structural conditions and successfully recognize special re- quirements(e.g.flat garment);
2) produce more stable texture and stripes;
3) make more reasonable inference even for the unseen parts;
4) obtain greater resemblance with fewer broken details. 

<img src="assets/Market.jpg"  width=256 height=384>

We also conduct experiments on Market-1501 datasets, which is a challenging dataset with low-resolution street person images. Our generated results [DNAF_Market_results](drive.google.com)

<img src="assets/otherApplication.jpg">
Further more, our model is capable of a series of downstream applications without extra fine-tuning:

1) Apperance transfer. Our model is capble of modifying the fashion style of a reference image while preserving other elements, making it ideal for virtual try-on applications.
2) Artwork Creation. Our model can also be applied to wild images and other forms of artwork. By replacing the style reference with artwork or a realistic photograph that may not necessarily feature a human subject, the model can easily copy the style and facilitate the generation of creative person images. This capability can be particularly beneficial for de- signers seeking inspiration.

## Method
<img src="assets/overview.jpg">

We narrow the feature-gap and imformation-gap. For more details please read our paper.

## ToDo
Once the paper is published, we will release the model and training code.

## Data Preparation
We evaluated our model on two public available dataset: DeepFashion and Market-1501. Moreover, we need to render the openpose extraced keypoints to skeleton-style pose map.

### DeepFashion

The DeepFash- ion dataset contains 52,172 high-resolution images of fash- ion models. Following previous works, we split the dataset into training set of 101,966 pairs and test set of 8,570 pairs. 
- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then rename the obtained folder as **img** and put it under the `./dataset/deepfashion` directory. 

- We split the train/test set following [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention). Several images with significant occlusions are removed from the training set. Download the train/test pairs and the keypoints `pose.zip` extracted with [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) by downloading the following files:

<!--   ```bash
  cd scripts
  ./download_dataset.sh
  ```

  Or you can download these files manually： -->

  - Download the train/test pairs from [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing) including **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Put these files under the  `./dataset/deepfashion` directory. 
  - Download the keypoints `pose.rar` extracted with Openpose from [Google Driven](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Unzip and put the obtained floder under the  `./dataset/deepfashion` directory.


### Market-1501
The Market- 1501 dataset is another important benchmark in the context of person re-identification. It consists of 32,668 low-resolution images captured in a street-style scenario. Training set con- tains 263,632 pairs while test set contains 12,000 pairs, and the identities for training and test does not overlap.

### OpenPose Map Render

## Model Preparation
Our work ultilized pretrained models for a basic image feature extraction ability and image generation ability to accelerating training.
### Stable Diffusion
```bash
git clone runwayml/stable-diffusion-v1-5
```
Other variant of diffusion model e.g. stable-diffusion-v1-4 stable-diffusion-v2-1 or stable diffusion-xl may also work well. Remove the model files to ```./checkpoints``` folder.
### Swin-Transformer
we ultilized swin-transformer-base for hierachical vision feature extraction.
```bash
git clone https://huggingface.co/microsoft/swin-base-patch4-window7-224
```
Other variants of swin-transformer or other hierachical vision encoders which output feature pyramid also acceptable. Remove the model files to ```./checkpoints``` folder.
## Training

## Inference


