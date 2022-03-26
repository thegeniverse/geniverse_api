import os
import gc
import logging
import subprocess
import time
from typing import List, Any, Tuple, Dict
from datetime import datetime
from threading import Thread

import torch
import torchvision
from PIL import Image
from upscaler.models import ESRGAN, ESRGANConfig
from geniverse.models import TamingDecoder
from geniverse_hub import hub_utils

# URL = "http://localhost:8008/"
URL = "http://34.255.194.217:8099/"


class GenerationManager:
    def __init__(
        self,
    ):
        self.generating = False
        self.generation_results_dict = {}

        self.user_queue = []
        self.current_user_id = ""

        self.generator = TamingDecoder(
            device="cuda",
            clip_model_name_list=[
                "ViT-B/32",
                "ViT-B/16",
            ],
        )

        self.upscaler = ESRGAN(
            ESRGANConfig(
                model_name="RealESRGAN_x4plus",
                tile=256,
            )
        )
        self.u2net = hub_utils.load_from_hub("u2net")

    def optimize(
        self,
        prompt_list: List[List[str]],
        prompt_weight_list: List[List[float]],
        num_iterations: int,
        resolution: Tuple[int],
        cond_img=None,
        device: str = "cuda",
        lr: float = 0.5,
        loss_type="spherical_distance",
        num_augmentations: int = 16,
        aug_noise_factor: float = 0.11,
        num_accum_steps: int = 8,
        init_step: int = 0,
        do_upscale: bool = False,
        results_dir: str = "results",
    ):
        # Pre-processing
        num_augmentations = max(1, int(num_augmentations / num_accum_steps))

        logging.debug(f"Using {num_augmentations} augmentations")
        logging.info(f"Using {num_accum_steps} accum steps")
        logging.info(f"Effective num crops of {num_accum_steps * num_augmentations}")

        assert loss_type in self.generator.supported_loss_types, (
            f"ERROR! Loss type "
            f"{loss_type} not recognized. "
            f"Only "
            f"{' or '.join(self.generator.supported_loss_types)} supported."
        )

        cond_img = cond_img.to(torch.float32)
        cond_img = cond_img.to(device)

        cond_img_size = cond_img.shape[2::]
        scale = (max(resolution)) / max(cond_img_size)
        img_resolution = [
            int((cond_img_size[0] * scale) // 16 * 16),
            int((cond_img_size[1] * scale) // 16 * 16),
        ]

        if scale != 1:
            cond_img = torch.nn.functional.interpolate(
                cond_img,
                img_resolution,
                mode="bilinear",
            )

        norm_cond_img = cond_img * 2 - 1
        norm_cond_img = torch.nn.functional.interpolate(
            cond_img,
            (200, 200),
            mode="bilinear",
        )

        # Foreground / Background detection
        mask = (
            self.u2net.get_img_mask(
                norm_cond_img,
            )
            .detach()
            .clone()
        )
        mask = torch.nn.functional.interpolate(
            mask,
            img_resolution,
            mode="bilinear",
        )
        mask = torchvision.transforms.functional.gaussian_blur(
            img=mask,
            kernel_size=[11, 11],
        )
        torchvision.transforms.ToPILImage()(mask[0]).save(
            f"{results_dir}/mask.png",
        )

        # Optimization step

        latents = self.generator.get_latents_from_img(cond_img)
        latents = latents.to(device)
        latents = torch.nn.Parameter(latents)
        optimizer = torch.optim.AdamW(
            params=[latents],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        for step in range(init_step, init_step + num_iterations):
            optimizer.zero_grad()

            logging.info(f"step {step}")

            def scale_grad(grad):
                grad_size = grad.shape[2:4]

                grad_mask = torch.nn.functional.interpolate(
                    mask,
                    grad_size,
                    mode="nearest",
                )

                grad.data = grad.data * grad_mask

                return grad

            img_rec_hook = latents.register_hook(
                scale_grad,
            )

            for _num_accum in range(num_accum_steps):
                loss = 0

                img_rec = torch.clip(
                    self.generator.get_img_from_latents(
                        latents,
                    ),
                    0,
                    1,
                )

                print(f"{latents.max()=}")
                print(f"{img_rec.max()=}")
                print(f"{mask.max()=}")
                print(f"{cond_img.max()=}")

                img_rec = img_rec * mask + cond_img * (1 - mask)

                if step == 0:
                    torchvision.transforms.ToPILImage()(cond_img[0]).save(
                        f"{results_dir}/{step:04d}.png",
                    )

                torchvision.transforms.ToPILImage()(img_rec[0]).save(
                    f"{results_dir}/{(step+1):04d}.png",
                )

                x_rec_stacked = self.generator.augment(
                    img_rec,
                    num_crops=num_augmentations,
                    noise_factor=aug_noise_factor,
                )

                img_logits_list = self.generator.get_clip_img_encodings(x_rec_stacked)

                for prompt, prompt_weight in zip(prompt_list, prompt_weight_list):
                    text_logits_list = self.generator.get_clip_text_encodings(prompt)

                    for img_logits, text_logits in zip(
                        img_logits_list, text_logits_list
                    ):
                        text_logits = text_logits.clone().detach()
                        if loss_type == "cosine_similarity":
                            clip_loss = (
                                -10
                                * torch.cosine_similarity(
                                    text_logits, img_logits
                                ).mean()
                            )

                        if loss_type == "spherical_distance":
                            clip_loss = (
                                (text_logits - img_logits)
                                .norm(dim=-1)
                                .div(2)
                                .arcsin()
                                .pow(2)
                                .mul(2)
                                .mean()
                            )

                        loss += prompt_weight * clip_loss

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            img_rec_hook.remove()

            torch.cuda.empty_cache()
            gc.collect()

        # Upscaling
        if do_upscale:
            # torchvision.transforms.ToPILImage()(img_rec[0]).save("results/test.png")
            img_rec = torch.clip(img_rec, 0, 1)
            img_rec = self.upscaler.upscale(img_rec).to(device, torch.float32) / 255.0

        return img_rec, latents

    def get_user_status(
        self,
        user_id: str,
    ):
        if user_id in self.user_queue:
            status = "Waiting"
        elif user_id in self.generation_results_dict.keys():
            status = "Done"
        elif user_id == self.current_user_id:
            status = "Generating"
        else:
            status = "Unknown"

        return status

    def get_user_results(
        self,
        user_id: str,
    ):
        result = None
        if user_id in self.generation_results_dict.keys():
            result = self.generation_results_dict.pop(user_id)

        return result

    def start_job(
        self,
        user_id: str,
        **kwargs,
    ):
        self.user_queue.append(
            user_id,
        )

        job_worker = Thread(
            target=self.generate_from_prompt,
            kwargs=kwargs,
        )
        job_worker.start()

        return

    def generate_from_prompt(
        self,
        prompt_list: str = "",
        num_nfts: int = 10,
        cond_img: Image.Image = None,
        auto: bool = True,
        param_dict={},
    ) -> List[Image.Image]:
        if auto:
            default_params = [
                {
                    "resolution": (512, 512),
                    "lr": 0.15,
                    "num_iterations": 10,
                    "do_upscale": False,
                    "num_crops": 128,
                },
                {
                    "resolution": (512, 512),
                    "lr": 0.1,
                    "num_iterations": 30,
                    "do_upscale": False,
                    "num_crops": 128,
                },
                {
                    "resolution": (640, 640),
                    "lr": 0.08,
                    "num_iterations": 20,
                    "do_upscale": True,
                    "num_crops": 64,
                },
            ]
            param_dict_list = default_params
        else:
            param_dict_list = [
                param_dict,
            ]

        try:
            generation_url_list = []
            video_url_list = []
            image_url_list = []
            while self.generating:
                time.sleep(5)
                logging.info("Already generating. Trying again in 5 seconds.")

            user_id = self.user_queue.pop()
            self.current_user_id = user_id

            self.generating = True

            timestamp = str(datetime.now()).split()
            prompts = ["_".join(prompt.split()) for prompt in prompt_list]
            filename_suffix = f"{'_'.join(timestamp)}-{'-'.join(prompts)}-{user_id}"
            results_dir = os.path.join("results", filename_suffix)
            os.makedirs(
                results_dir,
                exist_ok=True,
            )

            if cond_img is None:
                cond_img = Image.open("cosmic.png")

            cond_img = torchvision.transforms.PILToTensor()(
                cond_img,
            )[None, :]
            cond_img = cond_img / 255.0
            cond_img = cond_img.to("cuda", torch.float32)

            prompt_weight_list = [1 for _ in range(len(prompt_list))]

            for idx in range(num_nfts):
                init_step = 0
                for param in param_dict_list:
                    gen_img, _latents = self.optimize(
                        prompt_list=prompt_list,
                        prompt_weight_list=prompt_weight_list,
                        num_iterations=param["num_iterations"],
                        resolution=param["resolution"],
                        cond_img=cond_img,
                        device="cuda",
                        lr=param["lr"],
                        loss_type="cosine_similarity",
                        num_augmentations=param["num_crops"],
                        aug_noise_factor=0.11,
                        num_accum_steps=4,
                        init_step=init_step,
                        do_upscale=param["do_upscale"],
                        results_dir=results_dir,
                    )
                    init_step += param["num_iterations"]
                    cond_img = gen_img.detach().clone()

                gen_img_pil = torchvision.transforms.ToPILImage()(gen_img[0])

                iter_filename_suffix = f"{filename_suffix}_{idx}"

                image_path = f"results/{iter_filename_suffix}.png"
                gen_img_pil.save(image_path)

                fps = 10
                cmd = (
                    "ffmpeg -y "
                    "-r 8 "
                    f"-pattern_type glob -i '{results_dir}/0*.png' "
                    "-vcodec libx264 "
                    f"-crf {fps} "
                    "-pix_fmt yuv420p "
                    "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
                    f"results/{iter_filename_suffix}.mp4;"
                )

                subprocess.check_call(cmd, shell=True)

                generation_url_list.append(f"{URL}{iter_filename_suffix}")
                image_url_list.append(f"{URL}{iter_filename_suffix}.png")
                video_url_list.append(f"{URL}{iter_filename_suffix}.mp4")

            self.generation_results_dict[user_id] = {
                "generations": generation_url_list,
                "image": image_url_list,
                "video": video_url_list,
            }

        except Exception as e:
            print(f"ERROR {repr(e)}")

        finally:
            self.generating = False

        return iter_filename_suffix


if __name__ == "__main__":
    generation_manager = GenerationManager()
    nft_img_list = generation_manager.generate_from_prompt()
