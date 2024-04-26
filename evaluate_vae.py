import os
import torch
from vae.model import VAEModel
from vae.config import VAEConfig
from vae.dataloader import VAEDataLoader
from modules.run_evaluation import get_for_all_path

if __name__ == "__main__":

    configs = {
        "pilener_train": ["2024-04-23_11-17-43/model_5000"],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for dataset, model_paths in configs.items():
        for model_path in model_paths:
            model_ckpt = f"/vol/tmp/goldejon/gliner/vae/{dataset}/{model_path}"

            model_config = VAEConfig.from_pretrained(model_ckpt)

            dataloader = VAEDataLoader(
                token_encoder=model_config.token_encoder,
                label_encoder=model_config.label_encoder,
                dataloader_config=model_config.dataloader,
                device=device,
            )

            output_dir = f"/vol/tmp/goldejon/gliner/vae_eval/{dataset}/{model_path.split('/')[0]}"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model = VAEModel.from_pretrained(model_ckpt)
            model.to(device)
            model.eval()

            val_data_dir = "/vol/tmp/goldejon/gliner/eval_datasets/NER"
            get_for_all_path(
                model=model,
                log_dir=output_dir,
                data_paths=val_data_dir,
                dataloader=dataloader,
                only_zero_shot=True,
            )
