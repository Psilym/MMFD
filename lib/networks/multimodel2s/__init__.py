# from .whole_arch import FuseArch
from .whole_arch import  SimMIM_forMultiModelArch, MultiModelArchForSimMIM
from .fuse_arch import FusePre, FusePro
from .fuse_block import FusePro_Select, FuseFin_Select
from .build_seg import sam_model_registry