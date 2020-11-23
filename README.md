Name:Shruti Agrawal

Due Date: Sunday, November22, 2020  Author(s): Shruti Agrawal e-mail(s): sagrawa4@binghamton.edu

Following are the commands and the instructions to run the project.


## Pre Requisite.
- Python3 is needed to run the training and evalulation. 
- We have have dependency on numpy and scipy which can be downloaded as follows.
```
sudo pip3 install numpy
sudo pip3 install scipy
```


## Instructions to run:

Below command assumes `train.txt` file is present in current folder. Please download as its not 
present with source code tar.

```
python3 collaborative_filterting.py 
```

It take approx 10-15min to run above command, you should expect following output once finished.

```
Creating training and evaluation dataset......................................
Training the model, please wait, this can take few mins.......................
Evaluating the model, please wait, this can take few mins.....................
Root Mean Square Error is : 1.3171939872319491
```

It would also output following files.

`dataset.train` - Dataset used for training.
`dataset.eval` - Dataset used for evaluation.
`dataset.predict` - Final output.

If you want to run our own evaluation, you should be using data `dataset.predict`.


## Description:
We use collaborating filtering for training, for a given user we calculate similar user and for a given item we calculate similar items. When we need to predict rating we would than use similar user weighted average to predict rating, if we cannot predict rating than we would use similar items weighted average to predict rating, if we still cannot predict the rating its a cold start problem, we default to average rating(3).This project at the end returns a matrix that has all the predicted values populated for a 
user and item.


# Academic Honesty statement:
"I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating an official form will be submitted to the Academic Honesty Committee of the Watson School to determine the action that needs to be taken. "

Date: [11/22/2020]