import math
import random
import copy
from dataclasses import dataclass
from functools import cached_property

import gradio as gr

from backend import memory_management

from modules import errors, scripts
from modules.processing import (
    Processed,
    StableDiffusionProcessingTxt2Img,
    fix_seed,
    process_images,
)
from modules.script_callbacks import on_ui_settings
from modules.sd_models import model_data, select_checkpoint
from modules.shared import (
    OptionInfo, 
    opts, 
    state,
    total_tqdm,
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

    pixel_rounding: float = max(1, opts.data.get("arr_round_to", 64))

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
        (opts.data.get("arr_custom_ratios", "") or "").strip().split(",")
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

        total_ratios_to_generate = gr.Slider(
            minimum=1,
            maximum=1,
            step=1,
            value=1,
            label="Number of Ratios to Generate",
            info="Generate images using this many randomly selected aspect ratios"
        )

        def update_slider_max(selected_ratios, total_ratios):
            max_val = max(1, len(selected_ratios))
            current_val = min(total_ratios.value if hasattr(total_ratios, 'value') else 1, max_val)
            return gr.Slider(maximum=max_val, value=current_val)
        
        ratios.change(
            update_slider_max,
            inputs=[ratios, total_ratios_to_generate],
            outputs=[total_ratios_to_generate]
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

        return [ratios, total_ratios_to_generate]

    def run(self, p: StableDiffusionProcessingTxt2Img, ratios, total_ratios_to_generate):
        # Skip randomization if quick upscaling
        if hasattr(p, "txt2img_upscale") and p.txt2img_upscale:
            return process_images(p)

        if not ratios:
            errors.display(ValueError(
                "[Aspect Ratio Randomizer] Please select at least one aspect ratio"
            ))

        fix_seed(p)

        original_width = p.width
        iterations = p.n_iter * p.batch_size

        p.n_iter = 1
        p.batch_size = 1

        # Wildly different resolutions can make the grid image look weird
        p.do_not_save_grid = True

        selected_ratios = [ASPECT_RATIOS[ratio] for ratio in ratios]
        
        # Calculate total images needed
        total_images = iterations * total_ratios_to_generate
        
        # Create all processing objects
        processing_objects: list[StableDiffusionProcessingTxt2Img] = [p]
        
        # Create copies for all additional images
        for i in range(1, total_images):
            p_copy = copy.copy(p)
            # Seed is based on which iteration batch this belongs to
            iteration_num = i // total_ratios_to_generate
            p_copy.seed = p.seed + iteration_num
            processing_objects.append(p_copy)
        
        # Pre-generate random ratio selections for each iteration to avoid duplicates
        ratio_selections = []
        for iteration in range(iterations):
            if total_ratios_to_generate == len(selected_ratios):
                # Use all ratios in consistent order
                iteration_ratios = selected_ratios[:]
            elif total_ratios_to_generate <= len(selected_ratios):
                # Sample without replacement to avoid duplicates
                iteration_ratios = random.sample(selected_ratios, total_ratios_to_generate)
            else:
                # Need more ratios than available, sample with replacement
                iteration_ratios = random.choices(selected_ratios, k=total_ratios_to_generate)
            ratio_selections.extend(iteration_ratios)
        
        # Assign ratios to each processing object
        for idx, pc in enumerate(processing_objects):
            ratio = ratio_selections[idx]
                
            pc.width, pc.height = calc_nearest_res_for_ratio(original_width, ratio)

        hr_steps = p.hr_second_pass_steps if p.enable_hr else 0
        # Calculate total images that will be generated across all processing objects
        total_images = sum(pc.n_iter * pc.batch_size for pc in processing_objects)
        total_steps = sum(pc.n_iter * (pc.steps + hr_steps) * pc.batch_size for pc in processing_objects)

        state.job_count = total_images
        total_tqdm.updateTotal(total_steps)

        processed_result: Processed = None

        for idx, p in enumerate(processing_objects):
            memory_management.soft_empty_cache()

            if state.interrupted or state.stopping_generation:
                return Processed(p, [], p.seed, "")
            elif state.skipped:
                continue

            processed: Processed = None

            try:
                processed = process_images(p)
            except Exception as e:
                errors.display(e, "generating image with random aspect ratio")

            if processed is None:
                continue

            if processed_result is None:
                # Initialize with the first processed result
                processed_result = copy.copy(processed)
                processed_result.images = []
                processed_result.all_prompts = []
                processed_result.all_seeds = []
                processed_result.infotexts = []
                processed_result.index_of_first_image = 0

                # Update TQDM - other scripts may have changed final counts
                total_images *= len(processed.images)
                total_steps *= len(processed.images)

                state.job_count = total_images
                total_tqdm.updateTotal(total_steps)

            # Append ALL images and related data from this generation
            if processed.images:
                processed_result.images.extend(processed.images)
                processed_result.all_prompts.extend(processed.all_prompts)
                processed_result.all_seeds.extend(processed.all_seeds)
                processed_result.infotexts.extend(processed.infotexts)

            memory_management.soft_empty_cache()

        # Reorder the collections to group by ratio position rather than iteration
        if processed_result and total_ratios_to_generate > 1:
            # Calculate number of iterations
            num_iterations = len(processed_result.images) // total_ratios_to_generate
            
            # Create temporary lists to hold reordered data
            reordered_images = []
            reordered_prompts = []
            reordered_seeds = []
            reordered_infotexts = []
            
            # Reorder: instead of [1,1,1,1,2,2,2,2,3,3,3,3] we want [1,2,3,1,2,3,1,2,3,1,2,3]
            for ratio_idx in range(total_ratios_to_generate):
                for iter_idx in range(num_iterations):
                    source_idx = iter_idx * total_ratios_to_generate + ratio_idx
                    if source_idx < len(processed_result.images):
                        reordered_images.append(processed_result.images[source_idx])
                        reordered_prompts.append(processed_result.all_prompts[source_idx])
                        reordered_seeds.append(processed_result.all_seeds[source_idx])
                        reordered_infotexts.append(processed_result.infotexts[source_idx])
            
            # Replace the original collections with reordered ones
            processed_result.images = reordered_images
            processed_result.all_prompts = reordered_prompts
            processed_result.all_seeds = reordered_seeds
            processed_result.infotexts = reordered_infotexts

        checkpoint_info = select_checkpoint()

        model_data.forge_loading_parameters = dict(
            checkpoint_info=checkpoint_info,
            additional_modules=opts.forge_additional_modules,
            # unet_storage_dtype=opts.forge_unet_storage_dtype
            unet_storage_dtype=model_data.forge_loading_parameters.get(
                "unet_storage_dtype", None
            ),
        )

        return processed_result

section = ("arr", "Aspect Ratio Randomizer")


def on_settings():
    opts.add_option(
        "arr_custom_ratios",
        OptionInfo(
            None,
            "Custom Aspect Ratios",
            component=gr.Textbox,
            section=section,
        )
        .info(
            "Add custom aspect ratios to the list. Use the format 'width:height' separating entries with commas (,). No need to add the same ratio twice, ex: '3:4,4:3'."
        )
        .needs_reload_ui(),
    )

    opts.add_option(
        "arr_round_to",
        OptionInfo(
            64,
            "Pixel Rounding",
            gr.Slider,
            {"minimum": 0, "maximum": 128, "step": 32},
            section=section,
        ).info(
            "Round the calculated width and height to the nearest multiple of this number."
        ),
    )


on_ui_settings(on_settings)


print("Aspect Ratio Randomizer Loaded")
