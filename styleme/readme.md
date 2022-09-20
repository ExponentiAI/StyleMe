# StyelMe - pytorch
A pytorch implementation of image-to-sketch and sketch-to-image model.

## 1.0 Data
Include RGB image and sketch image of clothes in various styles.

## 1.1 Description
Related code comments:
* models.py: all the related models' structure definition, including encoder(style and content), decoder(decode random style features and content features), generator, and discriminator.
* datasets.py: data pre-processing and loading methods.
* encoder.py: AE module training.
* selfgan.py: GAN module training.
* config.py: all the hyper-parameters settings.
* benchmark.py: the FID functions, including inception model and it will automatically download. 
* lpips: the LPIPS functions, also including inception model and automatically download.
* train.py: training the hole model, and you can also choose train AE module only or train GAN module only.

## 1.2  Train
### 1.2.1 Train the image-to-sketch model
To train TOM for synthesizing sketches for an RGB-image dataset, put all your RGB-images in a folder, and place all you rcollected sketches into another folder
```
cd sketch_styletransfer
python train.py --path_a /path/to/RGB-image-folder --path_b /path/to/real-sketches
```
You can also see all the training options by:
```
python train.py --help
```
The code will automatically create a new folder to store all the trained checkpoints and intermediate synthesis results.

Once finish training, you can generate sketches for all RGB images in your dataset use the saved checkpoints:
```
python evaluate.py --path_content /path/to/RGB-image-folder --path_result /your/customized/path/for/synthesized-sketches --checkpoint /choose/any/checkpoint/you/like
```

### 1.2.2 Train the sketch-to-image model

Training the hole model such as:
```
python train.py 
```
## 1.3 Evaluate
You can transform the style:
```
python style_transform.py
```