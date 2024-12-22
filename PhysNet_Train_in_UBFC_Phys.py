# %%

from src.dataset_reader.UBFC_Phys import UBFCPhysDatasetReader
from src.data_generator.PhysNet import PhysNetDataGenerator,PhysNetDataConfig
from src.common.cache import CacheType
from src.stepbystep.v4 import StepByStep
from torch import optim
from src.model import PhysNet
from src.loss import Neg_PearsonLoss

MODEL_SAVE_PATH = r"./out/model/PhysNet_Train_in_UBFC-Phys.mdl"
TRAIN_CACHE = CacheType.READ
VAL_CACHE = CacheType.READ

dataset_path = r"/public/share/weiyuanwang/dataset/UBFC-Phys"
train_cache_path = r"~/cache/PhysNet/UBFC-Phys/train"
val_cache_path = r"~/cache/PhysNet/UBFC-Phys/val"

STEP = 70
T = 120
WIDTH = 128
HEIGHT = 128
BATCH = 2
dataset = 1


model = PhysNet(T)
loss = Neg_PearsonLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
sbs = StepByStep(model,loss,optimizer)


if __name__ == '__main__':
    train_dataset_config = PhysNetDataConfig(
        cache_root= train_cache_path,
        cache_type= TRAIN_CACHE,
        generate_num_workers=12,
        step=STEP,
        width=WIDTH,
        height=HEIGHT,
        slice_interval=T,
        num_workers = 12,
        batch_size=BATCH,
        shuffle=True,
        load_to_memory=False
    )
    val_dataset_config = PhysNetDataConfig(
        cache_root=val_cache_path,
        cache_type=VAL_CACHE,
        generate_num_workers=12,
        step=STEP,
        width=WIDTH,
        height=HEIGHT,
        slice_interval=T,
        num_workers=12,
        batch_size=BATCH,
        load_to_memory=False
    )
    train_dataset_reader = UBFCPhysDatasetReader(dataset_path=dataset_path,dataset=dataset,dataset_list=[
        's1', 's4', 's5', 's6', 's7', 's8','s11',
        's12','s13','s14','s15','s16','s17','s18','s19'
        's20','s23','s24','s25','s26','s27','s28','s29'
    ])
    val_dataset_reader = UBFCPhysDatasetReader(dataset_path=dataset_path,dataset=dataset,dataset_list=[
        's2', 's3', 's18','s19', 's21', 's22'
    ])
    train_dataset_generator = PhysNetDataGenerator(config=train_dataset_config)
    train_raw_data = train_dataset_reader.read() if TRAIN_CACHE == CacheType.NEW_CACHE else None
    train_dataloader = train_dataset_generator.generate(train_raw_data)

    val_dataset_generator = PhysNetDataGenerator(config=val_dataset_config)
    val_raw_data = val_dataset_reader.read() if VAL_CACHE == CacheType.NEW_CACHE else None
    val_dataloader = val_dataset_generator.generate(val_raw_data)

    sbs.set_loaders(train_dataloader,val_dataloader)
    sbs.set_tensorboard("physnet_train_in_UBFC-Phys")
    try:
        sbs.train(10)
    except:
        pass
    finally:
        sbs.save_best_checkpoint(MODEL_SAVE_PATH)
else:
    try:
        sbs.load_checkpoint(MODEL_SAVE_PATH)
    except:
        raise Exception('No any checkpoint, run train first!')