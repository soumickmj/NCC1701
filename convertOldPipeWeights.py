import os
from glob import glob
from tqdm import tqdm
from Bridge.WarpDrives.ReconResNet.ReconResNet import ResNet
from Engineering.utilities import ConvertCheckpoint

source_root = "/mnt/FCMn0301/data/project/SoumickPavan/NCC1701Weights/Enterprise_Gen3/Resnet2Dv2b14"
out_root = "/mnt/FCMn0301/data/project/SoumickPavan/NCC1701Weights/Enterprise_Gen3_Converted/ReconResNet2D14PReLU"

chkpaths = glob(f"{source_root}/**/*.pth.tar", recursive=True)

for chkpath in tqdm(chkpaths):
    try:
        out_chkpath = chkpath.replace(source_root, out_root)
        os.makedirs(os.path.dirname(out_chkpath), exist_ok=True)

        net = ResNet(in_channels=1, out_channels=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, 
                    do_batchnorm=False, res_drop_prob=0.2, is_replicatepad=0, out_act="sigmoid", 
                    forwardV=0, upinterp_algo='convtrans', post_interp_convtrans=False, is3D=False)  

        ConvertCheckpoint(checkpoint_path=chkpath, new_checkpoint_path=out_chkpath, newModel=net)
    except:
        pass

print("done!")     