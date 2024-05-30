# ai 4 in a row graphic interface:


## before running this file we need to train the model in my [google colab file](https://colab.research.google.com/drive/1rn8gKCMz4-O0brtKxHn1DPgfhIMUyRyc#scrollTo=Wyiw_YhBkWI4)

 after you had saved the weights in `your` drive in this line:

    Q_net.save_weights('/content/drive/MyDrive/model_DQN/DQN.keras')

download the file from your drive or conect your pc to drive and copy the filpath

## add the filpath into the code:
 first make sure the arcitecture if hte model is the same as in the google colab file 

 copy the filpath into line 90 of  `main_MK2`.py and make sure to keep the `r` before the filpath so it will read correctly. 
 
      link = r"INSERT FILPATH HERE"

also make sure the papameter `fesh_play = False` is set to `True` when you wish to play aginst the pre traind nural net.

# other papameters:

SQURESIZE = 125 - the size of the game

RADIUS = SQURESIZE/2 -3 - size of disk


## if you wish to change the size of the game change these parameters:
`NUM_COL = 7 `

`NUM_ROW = 6`
 
 also change the arcitecture of the nural net to fit the input and output shape.




 thanks to  `Gdebest69` [link](https://www.planetminecraft.com/member/gdebest69/)
 for helping with git