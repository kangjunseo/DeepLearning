## Implementation of S-Pred Paper

I got a lot of help of origin code (https://github.com/arontier/S_Pred_Paper).


There are 2 key points for this implementation.

First, I wrote lot of extra comments for studying, which is also written at paper, but not in origin code.  
This will be a big help for people who didn't read a paper carefully.  

Next, instead of arg-parsing program, I changed the code style to "just running".  
By changing the structure of program, I can understand more about this code, and also we can more easily
run a program in colab enviroment.

## Current issue
There is one terrible issue. In colab environment, there is no enough storage at GPU. According to paper,
original study was begun at NVIDIA Quadro RTX 8000 GPU, which has 48GB storage. Compare to that, colab has
much. smaller storage. I'm trying to search smaller msa transformer or smaller esm-msa.
