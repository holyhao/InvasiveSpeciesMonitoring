# InvasiveSpeciesMonitoring
- This is a competition held on kaggle which aims identifing images of invasive hydrangea. Further information at [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring)

## Information
- Tangles of kudzu overwhelm trees in Georgia while cane toads threaten habitats in over a dozen countries worldwide. These are just two invasive species of many which can have damaging effects on the environment, the economy, and even human health. Despite widespread impact, efforts to track the location and spread of invasive species are so costly that they’re difficult to undertake at scale. 
- Currently, ecosystem and plant distribution monitoring depends on expert knowledge. Trained scientists visit designated areas and take note of the species inhabiting them. Using such a highly qualified workforce is expensive, time inefficient, and insufficient since humans cannot cover large areas when sampling.
- Because scientists cannot sample a large quantity of areas, some machine learning algorithms are used in order to predict the presence or absence of invasive species in areas that have not been sampled. The accuracy of this approach is far from optimal, but still contributes to approaches to solving ecological problems.
- In this playground competition, Kagglers are challenged to develop algorithms to more accurately identify whether images of forests and foliage contain invasive hydrangea or not. Techniques from computer vision alongside other current technologies like aerial imaging can make invasive species monitoring cheaper, faster, and more reliable.

## Framework
- [Keras](https://keras.io/)
- [Tensorflow Backend](https://www.tensorflow.org/)

## Hardware
- Geforce GTX TITANX 12G
- Intel® Core™ i7-6700 CPU
- Memory 16G
- Operate system Ubuntu 14.04

## Data
- The data set contains pictures taken in a Brazilian national forest. In some of the pictures there is Hydrangea, a beautiful invasive species original of Asia. Based on the training pictures and the labels provided, the participant should predict the presence of the invasive species in the testing set of pictures.
- File descriptions
  - train.7z - the training set (contains 2295 images).
  - train_labels.csv - the correct labels for the training set.
  - test.7z - the testing set (contains 1531 images), ready to be labeled by your algorithm.
  - sample_submission.csv - a sample submission file in the correct format.
- Data fields
  - name - name of the sample picture file (numbers)
  - invasive - probability of the picture containing an invasive species. A probability of 1 means the species is present.
  
## Base Model
- [VGG16]() for deep feature extraction,which is provided in keras models.
- Softmax for classification.

## Evaluate
- For each image in the test set, you must predict a probability for the target variable on whether the image contains invasive species or not.

## to be continued
> Feel free to contact me if you have any issues or better ideas about anything.

> by Holy