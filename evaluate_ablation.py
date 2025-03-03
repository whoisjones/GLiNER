import os
import torch
import glob
from gliner import GLiNER
from gliner.modules.run_evaluation import get_for_all_path


if __name__ == "__main__":
    paths = glob.glob(
        "/vol/tmp/goldejon/gliner/logs_ablation_new_splits/nuner_train/*/*/*/model_20000"
    )
    for model_path in paths:
        metadata = model_path.split("/")
        dataset = metadata[-5]
        difficulty = metadata[-4]
        filter_by = metadata[-3]
        seed = metadata[-2]

        output_dir = f"/vol/tmp/goldejon/gliner/eval_ablation/{dataset}_v2/{difficulty}/{filter_by}/{seed}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = GLiNER.from_pretrained(model_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")

        val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
        get_for_all_path(
            model=model,
            log_dir=output_dir,
            data_paths=val_data_dir,
        )
