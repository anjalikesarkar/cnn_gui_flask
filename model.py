import torch
import torch.nn as nn

class classificationDemo(nn.Module):
    def __init__(self):
        super(classificationDemo, self).__init__()
        self.model = nn.Sequential(
                                    #1st conv layer
                                    nn.Conv2d(3,32,kernel_size=3), #i/p=(3,32,32),o/p=32,kernel=3 => (32,(32-3+1),(32-3+1))=>(32,30,30)
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),#i/p=(32,30,30)=>[(30-2)/2]+1=>15=>(32,15,15)
                                    
                                    #2nd conv layer
                                    nn.Conv2d(32,64,kernel_size=3),#i/p=(32,15,15),o/p=64,kernel=3=> (64,(15-3+1),(15-3+1))=>(64,13,13)
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),#i/p=(64,13,13)=>[(13-2)/2]+1= 6 =>(64,6,6)

                                    #3rd conv layer
                                    nn.Conv2d(64,64,kernel_size=3),#i/p=(64,6,6),o/p=64,kernel=3 => (64,(6-3+1),(6-3+1))=>(64,4,4)
                                    nn.ReLU(),

                                    #Flatten layer
                                    nn.Flatten(),
                                    nn.Linear(64*4*4,128), #i/p=(64,4,4),o/p=128
                                    nn.ReLU(),
                                    nn.Linear(128,10) #i/p=128, o/p=10(num_classes)
        )
    
    def forward(self,x):
        return self.model(x)
