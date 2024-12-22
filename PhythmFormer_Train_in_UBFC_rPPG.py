# %%
from src.dataset_reader.UBFC_rPPG import UBFCrPPGsDatasetReader
from data_generator.PhythmFormer import PhythmFormerDataGenerator,PhythmFormerDataGenerator
from src.common.cache import CacheType
from src.data_generator.LSTCrPPG import LSTCrPPGDataConfig
from src.dataset_reader.UBFC_rPPG import UBFCrPPGsDatasetReader
from src.common.cache import CacheType
from src.stepbystep.v4 import StepByStep
from torch import optim
from src.model import PhythmFormer
from src.loss import Neg_PearsonLoss,PhythmFormer_Loss
from torch import optim

USE_CACHED_UPPER_MODEL = False
MODEL_SAVE_PATH = r"./out/model/PhythmFormer_train_in_UBFC-rPPG.mdl"
TRAIN_CACHE = CacheType.READ
VAL_CACHE = CacheType.READ

dataset_path = r"/public/share/weiyuanwang/dataset/UBFC-rPPG"
train_cache_path = r"~/cache/PhythmFormer/UBFC-rPPG/train"
val_cache_path = r"~/cache/PhythmFormer/UBFC-rPPG/val"

STEP = 90
T = 160
WIDTH = 128
HEIGHT = 128
BATCH = 4

model = PhythmFormer()
loss = Neg_PearsonLoss()
# loss = PhythmFormer_Loss(30,1)
optimizer = optim.AdamW(model.parameters(), lr=9e-3, weight_decay=0)
# See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
sbs = StepByStep(model,loss,optimizer)

if __name__ == '__main__':
    train_dataset_config = LSTCrPPGDataConfig(
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
        load_to_memory=True
    )
    val_dataset_config = LSTCrPPGDataConfig(
        cache_root=val_cache_path,
        cache_type=VAL_CACHE,
        generate_num_workers=12,
        step=STEP,
        width=WIDTH,
        height=HEIGHT,
        slice_interval=T,
        num_workers=12,
        batch_size=BATCH,
        load_to_memory=True
    )
    train_dataset_reader = UBFCrPPGsDatasetReader(dataset_path,dataset=2,dataset_list=[
        'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15',
        'subject16', 'subject17', 'subject18', 'subject20', 'subject22', 'subject23', 'subject24',
        'subject25', 'subject26', 'subject27', 'subject30', 'subject31', 'subject32',
        'subject33', 'subject34', 'subject35', 'subject36', 'subject37', 'subject38', 'subject39',
        'subject40', 'subject41', 'subject42', 'subject43', 'subject44', 'subject45',
    ])
    val_dataset_reader = UBFCrPPGsDatasetReader(dataset_path,dataset=2,dataset_list=[
        'subject1','subject3','subject4','subject5','subject49', 
        'subject8', 'subject9', 'subject48','subject46', 'subject47',
    ])
    train_dataset_generator = PhythmFormerDataGenerator(config=train_dataset_config)
    train_raw_data = train_dataset_reader.read() if TRAIN_CACHE == CacheType.NEW_CACHE else None
    train_dataloader = train_dataset_generator.generate(train_raw_data)

    val_dataset_generator = PhythmFormerDataGenerator(config=val_dataset_config)
    val_raw_data = val_dataset_reader.read() if VAL_CACHE == CacheType.NEW_CACHE else None
    val_dataloader = val_dataset_generator.generate(val_raw_data)

    sbs.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=9e-3, epochs=30, steps_per_epoch=len(train_dataloader)//BATCH)
    sbs.set_loaders(train_dataloader,val_dataloader)
    sbs.set_tensorboard("PhythmFormer_Train_in_UBFC-rPPG")
    # try:
    sbs.train(20)
    # except:
    #     pass
    # finally:
    #     sbs.save_best_checkpoint(MODEL_SAVE_PATH)
else:
    try:
        sbs.load_checkpoint(MODEL_SAVE_PATH)
    except:
        raise Exception('No any checkpoint, run train first!')