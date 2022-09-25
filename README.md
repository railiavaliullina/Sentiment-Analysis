# Sentiment-Analysis

- task, the purpose of which is to determine the emotional coloring of the text (phrases, sentences) according to the input data.

## About The Project

This project contains:

    1) code for computing vector representations of reviews using ELMo (Embeddings from Language Models);
    2) implementation of RNN, CNN neural networks as baselines;
    3) implementation of Biattentive Classification Network + Maxout Network (BCN) from paper:

        McCann B. et al. Learned in translation: Contextualized word vectors, 2017

    4) implementation of Biattentive Classification Network + ReLU Feedforward Network (modification of BCN) from paper:

        Peters, Matthew E., et al. Deep contextualized word representations, 2018

    5) implementation of models training pipeline;
    6) comparison of all implemented models results.


## Dataset 
Movie reviews from IMdb:

    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  
## Implementation details

1) Computing vector representations of reviews using ELMo (Embeddings from Language Models) process visualization:

![image](https://user-images.githubusercontent.com/113569606/192167859-cea2cf4d-4a1c-46b3-b23c-c7e0d8f46df1.png)

2) 1st implemented model architecture:

![image](https://user-images.githubusercontent.com/113569606/192167938-3a940ea1-bf6c-4ac1-9c92-04a62e370ca4.png)

3) 2nd implemented model architecture:

![image](https://user-images.githubusercontent.com/113569606/192167949-bf8fb622-a9db-4a87-8c21-0b5f763af980.png)

4) 3rd implemented model architecture:

![image](https://user-images.githubusercontent.com/113569606/192167969-79c6c8ba-4e01-410e-8518-33d148ea10d1.png)

5) 4th implemented model architecture (Biattentive Classification Network + Maxout Network, paper: "Learned in translation: Contextualized word vectors"):

![image](https://user-images.githubusercontent.com/113569606/192167983-a7100c39-c08f-4520-9cf3-f19eb5fdfa35.png)

6) 5th implemented model architecture (Biattentive Classification Network + ReLU Feedforward Network, paper: "Deep contextualized word representations"):

        is same as previous one, except that ReLU activation function was used instead of Maxout after the last layer.

Maxout Function visualization:

![image](https://user-images.githubusercontent.com/113569606/192168123-b365359d-eacb-40c6-be72-dcbe39dd90b6.png)


# Training Results

![image](https://user-images.githubusercontent.com/113569606/192168170-3a61a281-54ce-4562-9fb5-beadc9cda8a0.png)

![image](https://user-images.githubusercontent.com/113569606/192168177-2f7df4c2-0fe4-424d-85d7-101c4ead83c8.png)

![image](https://user-images.githubusercontent.com/113569606/192168181-bad331e1-b3f6-42f3-9001-6e893b0e9c3d.png)

![image](https://user-images.githubusercontent.com/113569606/192168187-88612754-12d4-49bf-9572-d9071b1f9f54.png)

![image](https://user-images.githubusercontent.com/113569606/192168190-d178093d-615b-4980-b9b1-1528ad855b07.png)


# Best Model (according to obtained results)

## Architecture

![image](https://user-images.githubusercontent.com/113569606/192168248-8b9ce15e-952e-4b20-9023-bb7a4a42bb1a.png)

## Visualization of the training process

![image](https://user-images.githubusercontent.com/113569606/192168286-9758bc0b-1f98-44be-bc59-076625fcb988.png)
