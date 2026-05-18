import fire
from dataset_loader import get_dataset
from evaluation_loop import main_pipeline
from comparison import result_comparison

if __name__ == "__main__":
    fire.Fire({
        "get_dataset": get_dataset,
        "main_pipeline": main_pipeline,
        "result_comparison": result_comparison
    })



