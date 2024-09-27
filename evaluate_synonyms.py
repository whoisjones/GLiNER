import os

import pandas as pd
import torch

from evaluate_gliner import configs
from gliner import GLiNER
from gliner.modules.run_evaluation import get_for_all_path_with_synonyms

synonyms = {
    "person": ["person", "human", "individual", "somebody", "character"],
    "location": ["location", "place", "area", "spot", "site"],
    "organization": [
        "organization",
        "company",
        "institution",
        "association",
        "business",
    ],
    "event": ["event", "occurence", "happening", "incident", "occasion"],
}

if __name__ == "__main__":
    results = pd.DataFrame()

    output_dir = f"/vol/tmp/goldejon/gliner/eval_synonyms"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-large/{dataset}/{model_path}"
            )

            model = GLiNER.from_pretrained(model_ckpt)
            model.eval()
            if torch.cuda.is_available():
                model = model.to("cuda")

            val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
            result = get_for_all_path_with_synonyms(
                model=model,
                data_paths=val_data_dir,
                synonyms=synonyms,
            )
            result["train_dataset"] = dataset
            results = pd.concat([results, result], ignore_index=True)

    results.to_pickle(f"{output_dir}/results.pkl")
