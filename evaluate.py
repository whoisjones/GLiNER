from model import GLiNER
from modules.run_evaluation import get_for_all_path

if __name__ == "__main__":
    model = GLiNER.from_pretrained("urchade/gliner_small")
    model.eval()

    val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
    get_for_all_path(model, 0, "/vol/tmp/goldejon/gliner/test", val_data_dir)
