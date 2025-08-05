# MCS-UGAN
MCS-UGAN: Multiple Color Space Underwater GAN for Underwater Image Enhancement

## Before you start.
Clone Respository
```
git clone https://github.com/invincibility6/MCS-UGAN.git
```

## Training a Model
The training dataset should be placed in folder "./Dataset/train".
Start training by using the following command:
```  
python train.py
```

## Testing a Model
The testing dataset should be placed in folder "./Dataset/test".
Testing can be done using the following command by using your trained model or pre-trained model saved in "./checkpoints/UIEB/" directory:
``` 
python test.py --weights_path checkpoints/UIEB/generator_300.pth
```

## View the results
The operation result is generated in folder "./output".
