import os

import torch

from gliner import GLiNER
from gliner.modules.run_evaluation import get_for_all_path

if __name__ == "__main__":
    output_dir = f"/home/ec2-user/paper_data/eval/asknews/123"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1")
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    val_data_dir = "/home/ec2-user/paper_data/eval_datasets/NER"
    get_for_all_path(
        model=model,
        log_dir=output_dir,
        data_paths=val_data_dir,
    )
