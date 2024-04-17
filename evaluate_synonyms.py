import os
from save_load import load_model
from modules.run_evaluation import get_for_all_path

synonyms = {
    "person": ["person", "human", "individual", "people", "somebody"],
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
    configs = {
        "pilener_train": ["2024-03-12_14-40-13/model_30000"],
        "ontonotes": ["2024-03-12_17-05-12/model_30000"],
        "litset": ["2024-03-13_16-26-19/model_30000"],
        "fewnerd": ["2024-03-13_09-53-49/model_30000"],
    }

    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = (
                f"/vol/tmp/goldejon/gliner/logs/deberta-v3-small/{dataset}/{model_path}"
            )

            output_dir = f"/vol/tmp/goldejon/gliner/eval_synonyms/{dataset}/{model_path.split('/')[0]}"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model = load_model(model_ckpt)
            model.eval()

            val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
            get_for_all_path(
                model=model,
                log_dir=output_dir,
                data_paths=val_data_dir,
                only_zero_shot=True,
                synonyms=synonyms,
            )
