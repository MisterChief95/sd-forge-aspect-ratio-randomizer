import math
import random
import copy

import gradio as gr

from backend import memory_management
from modules.sd_models import model_data, select_checkpoint
import modules.shared as shared
from modules import errors, scripts
from modules.processing import (
    Processed,
    StableDiffusionProcessingTxt2Img,
    fix_seed,
    process_images,
)


ASPECT_RATIOS: dict = {
    "21:9": (21, 9),
    "16:9": (16, 9),
    "3:2": (3, 2),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "9:16": (9, 16),
    "9:21": (9, 21),
}


def calc_nearest_res_for_ratio(width: int, ratio: tuple[int, int]) -> tuple[int, int]:
    if ratio[0] == ratio[1]:
        return width, width

    base_area = width * width
    ratio = ratio[0] / ratio[1]

    if ratio > 1:
        # Scale width for positive ratios
        new_width = int(math.sqrt(base_area * ratio))
        new_height = int(new_width / ratio)
    else:
        # Scale height for negative ratios
        new_height = int(math.sqrt(base_area / ratio))
        new_width = int(new_height * ratio)

    new_width = round(float(new_width) / 64) * 64
    new_height = round(float(new_height) / 64) * 64

    return new_width, new_height


# TODO: Add settings options to allow users to add custom aspect ratios
class AspectRatioRandomizer(scripts.Script):
    def __init__(self):
        self.seed_to_ratio: dict[int, tuple[int, int]] = {}

    def title(self):
        return "Aspect Ratio Randomizer"

    def ui(self, is_img2img):
        if is_img2img:
            gr.Markdown(
                "This script is only available for text-to-image tasks. Please switch to Txt2Img tab to use this script."
            )
            return

        selector_mode = gr.Radio(
            value="Seed",
            choices=["Seed", "Random"],
            label="Ratio Selection Mode",
            info="Select whether to randomize aspect ratios based on seed or randomly",
        )

        ratios = gr.CheckboxGroup(
            label="Aspect Ratios",
            choices=ASPECT_RATIOS.keys(),
            info="Select the aspect ratios you want to randomize between. Order is: Wider -> Squarer -> Taller",
        )

        return [selector_mode, ratios]

    def run(self, p: StableDiffusionProcessingTxt2Img, selector_mode, ratios):
        if hasattr(p, "txt2img_upscale") and p.txt2img_upscale:
            return process_images(p)

        if not ratios:
            raise ValueError("Please select at least one aspect ratio")

        fix_seed(p)

        original_width = p.width
        iterations = p.n_iter * p.batch_size

        p.n_iter = 1
        p.batch_size = 1

        # Wildly different resolutions can make the grid image look weird
        p.do_not_save_grid = True

        processing_objects: list[StableDiffusionProcessingTxt2Img] = [p]

        if iterations > 1:
            for i in range(1, iterations):
                p_copy = copy.copy(p)
                p_copy.seed = p.seed + i
                processing_objects.append(p_copy)

        selected_ratios = [ASPECT_RATIOS[ratio] for ratio in ratios]

        for pc in processing_objects:
            if selector_mode == "Seed":
                ratio = selected_ratios[pc.seed % len(selected_ratios)]
            else:
                ratio = random.choice(selected_ratios)
            pc.width, pc.height = calc_nearest_res_for_ratio(original_width, ratio)

        hr_steps = p.hr_second_pass_steps if p.enable_hr else 0
        total_steps = len(processing_objects) * (p.steps + hr_steps)

        shared.state.job_count = len(processing_objects)
        shared.total_tqdm.updateTotal(total_steps)

        processed_result: Processed = None

        for idx, p in enumerate(processing_objects):
            memory_management.soft_empty_cache()

            if shared.state.interrupted or shared.state.stopping_generation:
                return Processed(p, [], p.seed, "")
            elif shared.state.skipped:
                continue

            processed: Processed = None

            try:
                processed = process_images(p)
            except Exception as e:
                errors.display(e, "generating image with random aspect ratio")

            if processed_result is None:
                img_count = len(processing_objects)
                processed_result = copy.copy(processed)
                processed_result.images = [None] * img_count
                processed_result.all_prompts = [None] * img_count
                processed_result.all_seeds = [None] * img_count
                processed_result.infotexts = [None] * img_count
                processed_result.index_of_first_image = 0

            if processed is None:
                continue

            if processed.images:
                processed_result.images[idx] = processed.images[0]
                processed_result.all_prompts[idx] = processed.prompt
                processed_result.all_seeds[idx] = p.seed
                processed_result.infotexts[idx] = processed.infotexts[0]

        memory_management.soft_empty_cache()

        checkpoint_info = select_checkpoint()

        model_data.forge_loading_parameters = dict(
            checkpoint_info=checkpoint_info,
            additional_modules=shared.opts.forge_additional_modules,
            # unet_storage_dtype=shared.opts.forge_unet_storage_dtype
            unet_storage_dtype=model_data.forge_loading_parameters.get(
                "unet_storage_dtype", None
            ),
        )

        return processed_result


print("Aspect Ratio Randomizer Loaded")
