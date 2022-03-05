from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

register_coco_instances("kucha_train", {}, "/home/erik/dhd-2022/beispiel-10/kucha_train.json", "/home/erik/dhd-2022/beispiel-06/images/")
register_coco_instances("kucha_test", {}, "/home/erik/dhd-2022/beispiel-10/kucha_test.json", "/home/erik/dhd-2022/beispiel-06/images/")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


cfg = get_cfg()
cfg.merge_from_file(
    "/home/erik/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("kucha_train",)
cfg.DATASETS.TEST = ("kucha_test",)
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = (
    50000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.TEST.EVAL_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.OUTPUT_DIR = "/home/erik/dhd-2022/beispiel-06/"
print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()