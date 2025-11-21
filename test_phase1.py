from aftershoot.src.config import get_config
from aftershoot.src.utils import say_hello
from aftershoot.src.dataloader import test_dataloader_import
from aftershoot.src.model import test_model_import
from aftershoot.src.losses import test_losses_import
from aftershoot.src.metrics import test_metrics_import
from aftershoot.src.train import test_train_script
from aftershoot.src.infer import test_infer_script

def main():
    print("CONFIG:", get_config())
    print("UTILS:", say_hello())
    print("DATALOADER:", test_dataloader_import())
    print("MODEL:", test_model_import())
    print("LOSSES:", test_losses_import())
    print("METRICS:", test_metrics_import())
    print("TRAIN:", test_train_script())
    print("INFER:", test_infer_script())
    print("\nPhase 1 import test completed successfully!")

if __name__ == "__main__":
    main()
