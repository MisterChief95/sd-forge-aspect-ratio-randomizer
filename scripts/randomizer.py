import math
import random
import copy
from dataclasses import dataclass
from functools import cached_property

import gradio as gr

from backend import memory_management

import modules.shared as shared
from modules.sd_models import model_data, select_checkpoint
from modules import errors, scripts
from modules.processing import (
    Processed,
    StableDiffusionProcessingTxt2Img,
    fix_seed,
    process_images,
)


DEFAULT_ASPECT_RATIOS: list[str] = ["21:9", "16:9", "3:2", "4:3", "1:1"]


@dataclass
class AspectRatio:
    antecedent: int
    consequent: int

    @cached_property
    def ratio(self) -> float:
        return self.antecedent / self.consequent


@dataclass
class Size:
    width: int
    height: int

    def __iter__(self):
        yield self.width
        yield self.height


def calc_nearest_res_for_ratio(width: int, aspect_ratio: AspectRatio) -> Size:
    if aspect_ratio.ratio == 1:
        return Size(width, width)

    base_area = width * width

    if aspect_ratio.ratio > 1:
        # Scale width for positive ratios
        new_width = int(math.sqrt(base_area * aspect_ratio.ratio))
        new_height = int(new_width / aspect_ratio.ratio)
    else:
        # Scale height for negative ratios
        new_height = int(math.sqrt(base_area / aspect_ratio.ratio))
        new_width = int(new_height * aspect_ratio.ratio)

    pixel_rounding: float = max(1, shared.opts.data.get("arr_round_to", 64))

    new_width = int(round(float(new_width) / pixel_rounding) * pixel_rounding)
    new_height = int(round(float(new_height) / pixel_rounding) * pixel_rounding)

    return Size(new_width, new_height)


def parse_aspect_ratio(ratio: str) -> tuple[str, AspectRatio]:
    ratio = ratio.replace(" ", "")
    antecedent, consequent = map(int, ratio.split(":"))
    key = f"{antecedent}:{consequent}"

    return key, AspectRatio(antecedent, consequent)


def reverse_ratio(ratio: str) -> str:
    antecedent, consequent = ratio.split(":")
    return f"{consequent}:{antecedent}"


def get_expanded_aspect_ratios() -> dict[str, AspectRatio]:
    custom_ratios = (
        (shared.opts.data.get("arr_custom_ratios", "") or "").strip().split(",")
    )
    custom_ratios = [
        ar.strip()
        for ar in custom_ratios
        if ":" in ar and ar.replace(":", "").isdigit()
    ]
    all_ratios = DEFAULT_ASPECT_RATIOS + custom_ratios
    expanded_ratios = all_ratios + [reverse_ratio(ratio) for ratio in all_ratios]

    return dict(
        sorted(
            [parse_aspect_ratio(ar) for ar in expanded_ratios],
            key=lambda item: item[1].ratio,
            reverse=True,
        )
    )


ASPECT_RATIOS: dict[str, AspectRatio] = get_expanded_aspect_ratios()
ratio_keys = list(ASPECT_RATIOS.keys())
IDX_1_1 = ratio_keys.index("1:1")
WIDE_RATIO_KEYS = ratio_keys[:IDX_1_1]
TALL_RATIO_KEYS = ratio_keys[IDX_1_1 + 1 :]


class AspectRatioRandomizer(scripts.Script):
    def title(self):
        return "Aspect Ratio Randomizer"

    def ui(self, is_img2img):
        if is_img2img:
            gr.Markdown(
                "This script is only available for text-to-image tasks. Please switch to Txt2Img tab to use this script."
            )
            return

        with gr.Row():
            ratios = gr.CheckboxGroup(
                label="Aspect Ratios",
                choices=list(ASPECT_RATIOS.keys()),
                info="Select the aspect ratios you want to randomize between. Order is: Wide - Square - Tall",
            )

        gr.HTML("<br>")

        with gr.Row():
            select_all = gr.Button(value="Select All")
            select_none = gr.Button(value="Select None")
        with gr.Row():
            select_wide = gr.Button(value="Select Wide")
            select_tall = gr.Button(value="Select Tall")
        with gr.Row():
            invert_select = gr.Button(value="Invert Selection")

        select_all.click(
            lambda _: gr.CheckboxGroup(value=list(ASPECT_RATIOS.keys())),
            inputs=[ratios],
            outputs=[ratios],
        )
        select_none.click(
            lambda _: gr.CheckboxGroup(value=[]), inputs=[ratios], outputs=[ratios]
        )
        select_wide.click(
            lambda _: gr.CheckboxGroup(value=WIDE_RATIO_KEYS),
            inputs=[ratios],
            outputs=[ratios],
        )
        select_tall.click(
            lambda _: gr.CheckboxGroup(value=TALL_RATIO_KEYS),
            inputs=[ratios],
            outputs=[ratios],
        )
        invert_select.click(
            lambda ratios: gr.CheckboxGroup(
                value=list(set(ASPECT_RATIOS) - set(ratios))
            ),
            inputs=[ratios],
            outputs=[ratios],
        )

        return [ratios]

    def run(self, p: StableDiffusionProcessingTxt2Img, ratios):
        # Skip randomization if quick upscaling
        if hasattr(p, "txt2img_upscale") and p.txt2img_upscale:
            return process_images(p)

        if not ratios:
            raise ValueError(
                "[Aspect Ratio Randomizer] Please select at least one aspect ratio"
            )

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
