import cv2  # oepnCV
import numpy as np
import torch
import torchvision
import PIL
import tqdm
import matplotlib

import sys
with open("version.txt", "w") as w:
    w.write(matplotlib.__name__ + " " + matplotlib.__version__ + "\n")
    w.write(cv2.__name__ + " "+cv2.__version__ + "\n")
    w.write(np.__name__ + " " + np.__version__ + "\n")
    w.write(torch.__name__ + " " + torch.__version__ + "\n")
    w.write(torchvision.__name__ + " " + torchvision.__version__ + "\n")
    w.write(PIL.__name__ + " " + PIL.__version__ + "\n")
    w.write(tqdm.__name__ + " " + tqdm.__version__ + "\n")
    w.write(sys.version + "\n")
    w.write("cuda " + torch.version.cuda + "\n")
    w.writelines(sys.modules.keys())