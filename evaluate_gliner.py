import os
import torch
from gliner import GLiNER
from gliner.modules.run_evaluation import get_for_all_path


configs = {
    # "ontonotes": ["123/model_60000", "234/model_60000", "345/model_50000"],
    # "fewnerd": ["123/model_60000", "234/model_60000", "345/model_60000"],
    "neretrieve_train": ["123/model_60000"]  # , "234/model_60000", "345/model_60000"],
    # "litset": ["123/model_60000", "234/model_60000", "345/model_60000"],
    # "nuner_train": ["123/model_60000", "234/model_60000", "345/model_60000"],
    # "pilener_train": ["123/model_60000", "234/model_60000", "345/model_60000"],
}

if __name__ == "__main__":
    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-large/{dataset}/{model_path}"
            )

            output_dir = (
                f"/vol/tmp/goldejon/gliner/eval/{dataset}/{model_path.split('/')[0]}"
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model = GLiNER.from_pretrained(model_ckpt)
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")

            val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
            get_for_all_path(
                model=model,
                log_dir=output_dir,
                data_paths=val_data_dir,
            )
