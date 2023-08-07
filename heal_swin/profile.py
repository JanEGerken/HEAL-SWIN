import json
from dataclasses import dataclass
import torch
from heal_swin.models_lightning import models_lightning
from heal_swin.data.data import get_data_module

import heal_swin.run_configs.segmentation.swin_hp_synwoodscape_large_plus_AD_train_run_config as syn_hp
import heal_swin.run_configs.segmentation.swin_synwoodscape_large_plus_AD_train_run_config as syn_baseline


@dataclass(frozen=True)
class ProfileConfig:
    batch_size: int = 1
    n_warmup: int = 10
    n_iter: int = 200


def instantiate_model_and_dataset(train_config):
    dm, data_spec = get_data_module(train_config.data)
    model_class = models_lightning.MODEL_FROM_CONFIG_NAME[train_config.model.__class__.__name__]
    model = model_class(
        config=train_config.model, data_spec=data_spec, data_config=train_config.data
    )
    model = model.cuda()

    return model, dm


def profile(run_config_module, profile_config):
    run_config = run_config_module.get_train_run_config()
    run_config.data.common.val_batch_size = profile_config.batch_size
    model, dm = instantiate_model_and_dataset(run_config)

    val_dataloader = dm.val_dataloader()

    images, _targets = next(iter(val_dataloader))
    images = images.cuda()

    print("Warming up...")
    for _ in range(profile_config.n_warmup):
        _warm_up = model(images)
    print("Profiling...")

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_times = []
    with torch.autograd.profiler.profile(use_cuda=True) as p:
        for idx in range(profile_config.n_iter):
            starter.record()
            _discard = model(images)
            ender.record()
            torch.cuda.synchronize()
            forward_times.append(starter.elapsed_time(ender))

    output_file_name = f"{run_config_module.__name__}.profile.txt"
    with open(output_file_name, "w") as output_file:
        output_file.write(run_config_module.__name__)
        output_file.write(f"\n[Data]\nImage shape: {images.shape}")
        output_file.write("\n[Profile config]\n")
        output_file.write(json.dumps(profile_config.__dict__, indent=2))
        output_file.write("\n[Profile results]\n")
        output_file.write(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        output_file.write("\n[Average GPU forward times]\n")
        output_file.write(
            f"mean forward: {torch.tensor(forward_times).mean()}, std forward: {torch.tensor(forward_times).std()}"
        )
    print(f"Wrote results to {output_file_name}")


if __name__ == "__main__":
    profile_config = ProfileConfig(batch_size=1, n_warmup=10, n_iter=200)
    profile(syn_hp, profile_config)
    profile(syn_baseline, profile_config)
