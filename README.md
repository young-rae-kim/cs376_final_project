# Birdcall Classifier using Mel-spectrogram and CNN
### CS376 Final Project Team 30
#### 20170131 Youngrae Kim, 20180767 SeongHyeok Kim, 20150837 ByeongHyeok Kim

  In biology, identifying bird species by sound is an important problem. There are some endangered bird species which biologists may not be able to capture and perform experiments, which makes it hard to collect experimental data. We thought if we construct a model that can predict which bird species we heard singing, we can aid biologists’ problems. Therefore, we would develop a machine learning model that can identify bird species with auditory data.

## 1. Data Set
* **Freefield1010** (for Model 1) : https://dcase.community/challenge2018/task-bird-audio-detection
* **BirdCLEF2021** (for Model 2) : https://www.kaggle.com/c/birdclef-2021 
* All audio data have been transformed into *Mel-spectrogram* images through the implemented pre-processing : <br/>https://github.com/young-rae-kim/cs376_final_project/blob/main/mel_transform.ipynb

## 2. Models
<img src="/model.jpg" alt="model"></img><br/>
* For CNN architectures deployed for each models, we use existing state-of-the-art CNN models as belows:
  * **ResNet50** : https://arxiv.org/abs/1512.03385 - *Selected based on experiments!*
  * **GoogLeNet** : https://arxiv.org/abs/1409.4842
  * **DenseNet** : https://arxiv.org/abs/1608.06993
* Two different models are deployed to detect bird species in given audio segment.
  * Model 1 (No-call Detector)
    *  indicates whether the audio has valid bird sound
    *  https://github.com/young-rae-kim/cs376_final_project/blob/main/nocall_detector.ipynb
  * Model 2 (Bird Species Classifier)
    *  indicates which bird species is likely to be included in the given audio
    *  https://github.com/young-rae-kim/cs376_final_project/blob/main/bird_species_classifier.ipynb

## 3. Train & Result
<img src="/result.jpg" alt="model"></img><br/>
* Model 1 (No-call Detector)
  * 10 epochs / lr = 1e-3, 𝝺 = 1e-3, momentum = 0.5
  * Final Accuracy on Test Set : **88.3%**
* Model 2 (Bird Species Classifier)
  * 25 epochs / lr = 5e-4, 𝝺 = 1e-3, momentum = 0.5
  * Final Accuracy on Test Set : **74.5%**

## 4. Demo
https://youtu.be/5AKESGzCdFw
