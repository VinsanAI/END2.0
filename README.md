# END_3_0_Assignment_2.5_pytorch

Assignment for Session 2.5 - Pytorch 101 (END3.0)

**Team Members :**
1) Santosh B H M - sbhm84@gmail.com
2) Ashutosh Kumar - ashutoshindian.ashu@gmail.com
3) Rajesh Kumar Birada - rajesh.bcool@gmail.com
4) Sateesh Ontikommu - sateesh.someswara@gmail.com

## 1) What is the task?
Task is to write a neural network that can take 2 inputs:
* An image from the MNIST dataset (say 5), and
* A random number between 0 and 9, (say 7)    

and it gives two outputs:
* the "number" that was represented by the MNIST image (predict 5), and
* the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

![image](https://user-images.githubusercontent.com/56379895/138541286-59e82655-3077-4ee1-8216-66b768e0d3e5.png)

We were successfully able to build this above mentioned network and you can find the notebook [here](https://github.com/VinsanAI/END3.0/blob/Assignment_2.5_Pytorch/END3_0_Assignment_2_5_pytorch.ipynb)

## 2) Data Representation :
Inputs :
* Image - size(1,28,28)
* Number - size(1,10)

Outputs from model :
* output_1 - Image_label - size(1,10)
* output_2 - Sum_label - size(1,19) - because values can range from 0-18

## 3) Data Generation Strategy :
We have generated the random numbers using numpy (np.random.randint) and we have created a function called 'create_one_hotencoding_inp' which will create the onehot encoding of the input number to the functiona nd outputs the lists of that encoded vector. 

So during data preparation we have created a new target called as 'sum_label' which is nothing but the sum of my random generated input numbers and the images labels already present with mnist. Thus we were successfully able to create the new label for my network.

## 4) How did we combine two inputs?

![image](https://user-images.githubusercontent.com/56379895/138541859-23d88eb6-809c-4bbe-8857-cfffb21aef69.png)

In the above image as you can see after we do the convolutions on the image when we flatten all the values to fully connected network we are adding just 10 nodes at this place for which we will pass the input of random number during training.

## 5) how you are evaluating your results ?
Model evaluation is done at each epoch by calculating the percentage of correct labels predicted and we observe whether our loss is reducing over the epoch if not we have tried changing multiple parameters like increasing trainset, changing batch size, changing learning rate etc
below are the train accuracies : 
* image label : 99.9%
* sum label : 82.03%

Once the training is completed we have also built the inference block where we will send in some random image and random number, we expect model to predict the image label and sum label.

## 6) What are the results you got?
Our model is outputting two vectors
output_1 - Image label - size(1,10)
output_2 - Sum Label - size(1,18)

We have got the arg max of the each of the output as them being softmax outputs hence they are prabability values, so argmax will give us the label with highest probability score.

## 7) What loss function you picked?
We ave picked the l1 loss which is l1_loss = L1+L2, we just add up both the losses and we try to optimize on this total loss.
