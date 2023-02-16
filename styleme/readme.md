# Environment :
python 3.8.0 pytorch 1.12.1

## 1. Description
Related code comments:

* train.py: training the hole model, and you can also choose train AE module only or train GAN module only.
* models.py: all the related models' structure definition, including encoder(style and content), decoder(decode random style features and content features), generator, and discriminator.
* datasets.py: data pre-processing and loading methods.
* train_step_1.py: AE module training.
* train_step_2.py: GAN module training.
* config.py: all the hyper-parameters settings.
* calcualte.py: calculate the FID and LPIPS of the model.
* benchmark.py: the FID functions, including inception model and it will automatically download.
* lpips: the LPIPS functions, also including inception model and automatically download.
* style_transform.py: put your sketch and RGB images to tansform the style.


## Training

first prepare your datasets as follows:

```
train_data/
  -./rgb/
    -000.png
    -001.png
    -...
  -./sketch/
    -000.png
    -001.png
    -...
```

and then training your models:

```
python train.py 
```

## Evaluate

You can run the following program to see the performance of our model:

```
python style_transform.py 
```

or you can also get the FID and LPIPS:

```
python calculate.py 
```