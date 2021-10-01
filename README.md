
# END_3_0_Assignment_Session_1

Assignment for Session 1 - Background & Very Basics (END3.0)

**Team Members :**
1) Santosh B H M - sbhm84@gmail.com
2) Ashutosh Kumar - ashutoshindian.ashu@gmail.com
3) Rajesh Kumar Birada - rajesh.bcool@gmail.com
4) Sateesh Ontikommu - sateesh.someswara@gmail.com


## 1)	What is a neural network neuron?
> It’s a building block of a neural network which stores a value temporarily. It can contain input value or processed non-linear data(Activated value).

## 2)	What is the use of the learning rate?
> In Neural Networks the Learning Rate (LR) is a hyperparameter which helps the model to converge to a local minima and also represents how quickly my model can get adapted to the problem. Tuning the model to right LR is a very important task as keeping it very low might lead to higher training time but where as keeping high LR may lead to improper convergence.

![s1](https://user-images.githubusercontent.com/56379895/134590343-11b601a3-0a5c-45f0-8381-4da46832e114.jpg)

> Let’s assume that in the above equation all the weights are influencing each other and now we are calculating new updated weight for w2, in this scenario the confusion will be that whether we should consider the old w1 value or the new updated weight of w1. And if we keep considering the new updated values every time when we compute new values for subsequent weights, then we will be stuck in an infinite loop. 

![s2](https://user-images.githubusercontent.com/56379895/134590474-9ffc0402-3db6-4f42-9b55-b1d31e0a9ab7.jpg)

> Hence to resolve this issue partial derivatives comes in to picture, where we will assume that while computing update for a specific weight, all the other weights remain constant. And while doing so before updating any weight we multiply the partial derivatives (gradients) with a smaller value called Learning Rate (LR) because even if there was some kind of impact from w1 on w2, this LR will make the updates so small that the influence of w1 on w2 will become mathematically insignificant.

## 3)	How are weights initialized?
> Generally weights are initialized randomly between 0 & 1 Because initializing all the weights to same value or to 0.0 will result in some kind of asymmetry in the error gradient which will begin searching ineffectively. Consider the below image where we are assigning different set of weights which are not under same scale and if we look at the outputs after all the computations they are way apart from each other, because 0.00001 is just like a pinch of salt in the ocean of 10xe13. So calculating gradients with this magnitude will not make sense.

![image](https://user-images.githubusercontent.com/56379895/134678208-d6227f71-34b7-4e37-92b5-edc9fe171474.png)

> Hence weights are normalized to be under one scale and mostly be nearest to zero because this will make sure that the feedback from backpropagation will make sense for all the values computed in the hidden layers while computing gradients.

## 4)	What is "loss" in a neural network?
> Loss is the difference between the predicted and the actual output value in the output layer. Our main objective in a neural network during backpropagation is to reduce the loss over multiple epocs which leads to adjust right weights there by making accurate predictions.

## 5)	What is the "chain rule" in gradient flow?
> Chain Rule is something which enables the backpropagation to happen effectivity by considering the contribution from the neurons coming in between. 

![s3](https://user-images.githubusercontent.com/56379895/134590501-30e98785-07cc-4362-a312-e13b003e5202.jpg)

> For expample consider the below drawn simple neural network and we are calculating the gradient for weight w111, In this case according to chain rule we should also consider the contribution from the weight w311 which is coming in between while backpropagating.

![image](https://user-images.githubusercontent.com/56379895/134678064-c6d7539e-0143-4855-8b41-6cc33f11f809.png)

  
