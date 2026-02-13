import os
import numpy as np


n = 64
cwd = os.getcwd()
data_dir = os.path.join(cwd, "fmaps", f"model{n}_fmaps")
rows = []
file_list = sorted(os.listdir(data_dir))

for i,file in enumerate(file_list):
    cls = str(file[6:7]) ### this is the position where the index of the class is
    initLength = len(str(file_list[0]))
    currFile = file_list[i]
    currLength = len(str(currFile))
    if currLength > initLength:
        idx = str(currFile[12:14])
    else:
        idx = str(currFile[12:13])

    fmaps = np.load(os.path.join(data_dir, str(currFile)))
    alpha = fmaps["alpha"]
    img = fmaps["image"]
    fm = fmaps["feature_maps"]
    gradients = fmaps["gradients"]
    grad_cams = fmaps["grad_cams"]
    rows.append({"Class": cls, "Idx": idx, "Image": img,  "Grad-contribution": alpha, "FMaps": fm, "Gradients": gradients, "Grad-cam": grad_cams})


np.savez(os.path.join(os.path.join(os.getcwd(), "dicts"), f"dataframe_model{n}.npz"), rows=np.array(rows, dtype=object))