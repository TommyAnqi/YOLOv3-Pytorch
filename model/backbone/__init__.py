from . import darknet
from . import mobilenetv2
import pdb


def backbone_fn(opt):
    if opt.backbones_network == "darknet21":
        model = darknet.darknet21(opt.weightfile)
    elif opt.backbones_network == "darknet53":
        model = darknet.darknet53(opt.weightfile)
    elif opt.backbones_network == "mobilenetv2":
        model = mobilenetv2.mobilenetv2(opt.weightfile)
    elif opt.backbones_network == "modified_mobilenetv2":
        model = mobilenetv2.mobilenetv2(opt.weightfile)
    else:
        pdb.set_trace()
    return model