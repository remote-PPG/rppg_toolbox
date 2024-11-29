# %%
from src.dataset_reader.UBFC_rPPG import UBFCrPPGsDatasetReader
from src.data_generator.LSTCrPPG import LSTCrPPGDataGenerator
from src.common.cache import CacheType
from src.data_generator.LSTCrPPG import LSTCrPPGDataConfig
from src.dataset_reader.UBFC_rPPG import UBFCrPPGsDatasetReader
from src.common.cache import CacheType
from src.stepbystep.v4 import StepByStep
from torch import optim
from src.model import LSTCrPPG
from src.loss import LSTCrPPGLoss
from torch.nn import MSELoss
# from .loss import Neg_PearsonLoss


USE_CACHED_UPPER_MODEL = False
MODEL_SAVE_PATH = r"./out/model/LSCTrPPG_train_in_UBFC-rPPG.mdl"
TRAIN_CACHE = CacheType.READ
VAL_CACHE = CacheType.READ

dataset_path = r"/public/share/weiyuanwang/dataset/UBFC-rPPG"
train_cache_path = r"~/cache/LSCTrPPG/UBFC-rPPG/train"
val_cache_path = r"~/cache/LSCTrPPG/UBFC-rPPG/val"

STEP = 70
T = 160
WIDTH = 128
HEIGHT = 128
BATCH = 2

model = LSTCrPPG()
loss = LSTCrPPGLoss()
# loss = Neg_PearsonLoss()
optimizer = optim.Adam(model.parameters(),lr=5e-5)
sbs_lstc_rppg = StepByStep(model,loss,optimizer)


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
    train_dataset_generator = LSTCrPPGDataGenerator(config=train_dataset_config)
    train_raw_data = train_dataset_reader.read() if TRAIN_CACHE == CacheType.NEW_CACHE else None
    train_dataloader = train_dataset_generator.generate(train_raw_data)

    val_dataset_generator = LSTCrPPGDataGenerator(config=val_dataset_config)
    val_raw_data = val_dataset_reader.read() if VAL_CACHE == CacheType.NEW_CACHE else None
    val_dataloader = val_dataset_generator.generate(val_raw_data)

    sbs_lstc_rppg.set_loaders(train_dataloader,val_dataloader)
    sbs_lstc_rppg.set_tensorboard("LSTCrPPG_Train_in_UBFC-rPPG")
    try:
        sbs_lstc_rppg.train(40)
    except:
        pass
    finally:
        sbs_lstc_rppg.save_best_checkpoint(MODEL_SAVE_PATH)
else:
    try:
        sbs_lstc_rppg.load_checkpoint(MODEL_SAVE_PATH)
    except:
        raise Exception('No any checkpoint, run train first!')