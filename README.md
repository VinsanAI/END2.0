# END_3_0_Assignment_Session_2

Assignment for Session 2 - Backpropagation, Embeddings &amp; Language Models

**Team Members :**
1) Santosh B H M - sbhm84@gmail.com
2) Ashutosh Kumar - ashutoshindian.ashu@gmail.com
3) Rajesh Kumar Birada - rajesh.bcool@gmail.com
4) Sateesh Ontikommu - sateesh.someswara@gmail.com

# 1. Below is the screenshot of the excel file uploaded as part of the repository :

![image](https://user-images.githubusercontent.com/56379895/135353050-3b183eb3-2b79-4b70-932b-1345a49bee09.png)

# 2. Below is the GIF for error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] :

![ezgif com-gif-maker](https://user-images.githubusercontent.com/56379895/135417225-2fd0d254-0f99-44bf-a6d2-0802d6e00788.gif)

> **Note :** We have also added the screenshots of the above gif in the repository for reference, please find them [here](https://github.com/VinsanAI/END_3_0_Assignment_Session_2/tree/main/Change%20in%20Error%20with%20LR%20-%20Screenshots).

# 3. Detailed Explaination of each major step :

![image](https://user-images.githubusercontent.com/56379895/135442691-96b68e69-79a4-4beb-bd42-d0cebe4fa6d1.png)

Lets assume we have the neural network defined in the above diagram where
* i1 & i2 are input nurons
* o1 & o2 are output neurons
* h1 & h2 are intermediate neurons which are part of hidden layer
* E1 & E2 are the errors for o1 & o2 classes respectively, where as E_t is the combined error
* w1...w8 are the weights that are connecting the nodes with each other

As we already know that :   
`h1 = i1*w1 + i2*w2`   
`h2 = i1*w3 + i2*w4`   
`a_h1 = σ(h1) = 1/(1 + exp(-h1))`   
`a_h2 = σ(h2) = 1/(1 + exp(-h2))`   
`o1 = a_h1*w5 + a_h2*w6`   
`o2 = a_h1*w7 + a_h2*w8`   
`a_o1 = σ(o1)`   
`a_o2 = σ(o2)`   
`E1 = ½ * (t1 - a_o1)2`   
`E2 = ½ * (t2 - a_o2)2`

### Calculating Gradients for `w5`, `w6`, `w7` & `w8` :
**Lets now calculate the gradients for w5 :**   
`∂E_t/∂w5 = (∂E1 + ∂E2)/∂w5 = ∂E1/∂w5 = (∂E1/∂a_o1)*(∂a_o1/∂o1)*(∂o1/∂w5)`   
> **Note :** `∂E2/∂w5=0` because `E2` has no relation with `w5`   

**Now lets calculate `∂E1/∂a_o1`,`∂a_o1/∂o1` and `∂o1/∂w5` these guys separately**,   
`∂E1/∂a_o1  = ∂(½ * (t1 - a_o1)2)/∂a_o1 = ½*2*(t1-a_01)*(-1) = (a_o1 - t1)`   
`∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = σ(o1)(1 - σ(o1)) = a_o1*(1-a_o1)`
> **Note :** `∂(σ(x))/∂(x)  = σ(x)*(1-σ(x))` 
  
`∂o1/∂w5 = ∂(a_h1*w5 + a_h2*w6)/∂w5 = a_h1`   

**Hence gradients for `w5, w6, w7 & w8` will be :**   
**`∂E_t/∂w5 = (a_o1 - t1) * a_o1 * (1 - a_01) * a_h1`**   
**`∂E_t/∂w6 = (a_o1 - t1) * a_o1 * (1 - a_01) * a_h2`**       
**`∂E_t/∂w7 = (a_o2 - t2) * a_o2 * (1 - a_02) * a_h1`**   
**`∂E_t/∂w8 = (a_o2 - t2) * a_ o2 * (1 - a_02) * a_h2`**   

### Calculating Gradients for `w1`, `w2`, `w3` & `w4` :
**lets calculate the gradients for `w1` :**   
`∂E_t/∂w1 = ∂E_t/∂a_01 * ∂a_01/∂o1 * ∂o1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1`   
`∂E_t/∂w1  = ∂E_t/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1`   
`∂E_t/∂w1 = ∂E_t/∂a_h1 * ∂(σ(h1))/∂h1 * ∂h1/∂w1 = ∂E_t/∂a_h1 * a_h1 * (1 - a_h1) * ∂h1/∂w1`

`∂E_t/∂w1 = ∂E_t/∂a_h1 * a_h1*(1 - a_h1) * i1`   
`∂E_t/∂w2 = ∂E_t/∂a_h1 * a_h1*(1 - a_h1) * i2`   
`∂E_t/∂w3 = ∂E_t/∂a_h2 * a_h2*(1 - a_h2) * i1`   
`∂E_t/∂w4 = ∂E_t/∂a_h2 * a_h2*(1 - a_h2) * i2`   

**Now lets calculate `∂E_t/∂a_h1` :**    
`∂E_t/∂a_h1 = ∂(E1+E2)/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1`   

`∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1 = (a_o1 - t1) * a_o1*(1-a_o1) * w5`   
`∂E2/∂a_h1 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h1  = (a_o2 - t2) * a_o2*(1-a_o2) * w7`   

`∂E1/∂a_h2 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h2 = (a_o1 - t1) * a_o1*(1-a_o1) * w6`   
`∂E2/∂a_h2 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o2/∂a_h2 = (a_o2 - t2) * a_o2*(1-a_o2) * w8`   

`∂E_t/∂a_h1 = (a_o1 - t1) * a_o1*(1-a_o1) * w5 + (a_o2 - t2) * a_o2*(1-a_o2) * w7`   
`∂E_t/∂a_h2 = (a_o1 - t1) * a_o1*(1-a_o1) * w6 + (a_o2 - t2) * a_o2*(1-a_o2) * w8`   

**So after substitution we will get the final gradienst for `w1`, `w2`, `w3` & `w4` :**   
**`∂E_t/∂w1 = ((a_o1 - t1) * a_o1*(1-a_o1) * w5 + (a_o2 - t2) * a_o2*(1-a_o2) * w7) * a_h1*(1 - a_h1) * i1`**   
**`∂E_t/∂w2 = ((a_o1 - t1) * a_o1*(1-a_o1) * w5 + (a_o2 - t2) * a_o2*(1-a_o2) * w7) * a_h1*(1 - a_h1) * i2`**    
**`∂E_t/∂w3 = ((a_o1 - t1) * a_o1*(1-a_o1) * w6 + (a_o2 - t2) * a_o2*(1-a_o2) * w8) * a_h2*(1 - a_h2) * i1`**   
**`∂E_t/∂w4 = ((a_o1 - t1) * a_o1*(1-a_o1) * w6 + (a_o2 - t2) * a_o2*(1-a_o2) * w8) * a_h2*(1 - a_h2) * i2`**   
