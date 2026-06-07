import math
import random
from dataclasses import dataclass
from functools import cached_property

import gradio as gr

from modules import errors, scripts
from modules import processing as processing_module
from modules import rng as rng_module
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import on_ui_settings
from modules.shared import (
    OptionInfo,
    opts,
)
from modules.ui_components import InputAccordion


DEFAULT_ASPECT_RATIOS: list[str] = ["21:9", "16:9", "3:2", "4:3", "1:1"]
RATIO_SELECTION_MODES: list[str] = [
    "Random, no repeats",
    "Selected order",
    "Sequential rotating",
    "Random with replacement",
]

LOG = "[Aspect Ratio Randomizer]"


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

    # Follow the WebUI's built-in "Resolution Step" so computed sizes stay valid.
    pixel_rounding: float = max(1, opts.data.get("res_step", 64))

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
    raw = (opts.data.get("arr_custom_ratios", "") or "").split(",")
    # strip, drop empties, dedupe while preserving order
    custom_ratios = list(dict.fromkeys(ar.strip() for ar in raw if ar.strip()))
    valid_custom_ratios: list[str] = []

    for ar in custom_ratios:
        parts = ar.split(":")
        if (
            len(parts) == 2
            and parts[0].isdigit()
            and parts[1].isdigit()
            and int(parts[0]) > 0
            and int(parts[1]) > 0
        ):
            valid_custom_ratios.append(ar)
        else:
            print(f"{LOG} ignoring invalid custom aspect ratio: '{ar}'")

    all_ratios = DEFAULT_ASPECT_RATIOS + valid_custom_ratios
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
    section = "dimensions"
    create_group = False
    sorting_priority = 15

    def title(self):
        return "Aspect Ratio Randomizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        with InputAccordion(
            False, label=self.title(), elem_id="arr-enabled"
        ) as enabled:
            gr.Markdown(
                "**Batch count** controls comparison groups. "
                "**Batch size** is the maximum GPU parallelism."
            )
            gr.Markdown(
                "The **Width** setting is used as a base size to compute the others from, "
                "so pick one that works for all your ratios (e.g. 1024 for SDXL)."
            )

            variants_per_ratio = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=1,
                label="Variants per ratio",
                info="Number of seed / prompt variants to render for each ratio in each comparison group",
            )

            ratios_per_batch = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=1,
                label="Ratios per batch",
                info="Number of selected aspect ratios to render for each comparison group",
            )

            with gr.Row():
                selection_mode = gr.Dropdown(
                    choices=RATIO_SELECTION_MODES,
                    value=RATIO_SELECTION_MODES[0],
                    label="Ratio selection",
                )

                sort_gallery = gr.Checkbox(
                    value=True,
                    label="Sort gallery by seed then resolution",
                )

            with gr.Row():
                ratios = gr.CheckboxGroup(
                    label="Aspect Ratios",
                    choices=list(ASPECT_RATIOS.keys()),
                    info="Select the aspect ratios you want to randomize between. Order is: Wide - Square - Tall",
                )

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

        return [enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery]

    def before_process(self, p, enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery):
        """Configure Forge's prompt/seed arrays for exact comparison groups.

        ``batch count`` is treated as the number of comparison groups, while
        ``variants_per_ratio`` controls how many seed/prompt variants are created
        per group. ``batch size`` stays a maximum: we choose the largest exact
        chunk size that divides ``variants_per_ratio`` and does not exceed the
        user's requested batch size.
        """
        p._arr_active = False

        if not enabled:
            return
        if not isinstance(p, StableDiffusionProcessingTxt2Img):
            return  # img2img init image would be distorted
        if getattr(p, "txt2img_upscale", False):
            return  # hires "quick upscale" button: leave dimensions alone

        if not ratios:
            errors.display(
                ValueError(
                    "[Aspect Ratio Randomizer] Please select at least one aspect ratio"
                )
            )
            return

        selected = [ASPECT_RATIOS[r] for r in ratios if r in ASPECT_RATIOS]
        if not selected:
            return

        p._arr_active = True
        p._arr_base_w = p.width
        p._arr_selected = selected
        p._arr_group_count = max(1, int(p.n_iter))
        p._arr_variants_per_ratio = max(1, int(variants_per_ratio or 1))
        p._arr_requested_batch_size = max(1, int(p.batch_size))
        p._arr_chunk_size = self._largest_divisor_at_most(
            p._arr_variants_per_ratio,
            p._arr_requested_batch_size,
        )
        p._arr_chunks_per_group = p._arr_variants_per_ratio // p._arr_chunk_size
        p._arr_ratios_per_batch = max(1, int(ratios_per_batch or 1))
        p._arr_selection_mode = selection_mode if selection_mode in RATIO_SELECTION_MODES else RATIO_SELECTION_MODES[0]
        p._arr_sort_gallery = bool(sort_gallery)
        p._arr_plan = []
        p._arr_overflow_warned = False
        p._arr_gallery_sort_keys = []

        p.batch_size = p._arr_chunk_size
        p.n_iter = p._arr_group_count * p._arr_chunks_per_group

        if p._arr_chunk_size != p._arr_requested_batch_size:
            print(
                f"{LOG} using GPU batch {p._arr_chunk_size} instead of requested "
                f"{p._arr_requested_batch_size} so {p._arr_variants_per_ratio} "
                f"variant(s) per ratio stay exact"
            )

    def process(self, p, enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery):
        """Repeat each comparison group's source variants across selected ratios.

        This hook runs after ``setup_prompts()`` and after Dynamic Prompts when the
        extension metadata loads ARR after sd-dynamic-prompts. We duplicate each
        already-expanded prompt/seed group across ratios, keeping the exact
        internal chunk size chosen in ``before_process``.
        """
        if not getattr(p, "_arr_active", False):
            return

        selected = getattr(p, "_arr_selected", [])
        if not selected:
            return

        batch_size = max(1, int(p.batch_size))
        group_count = max(1, int(getattr(p, "_arr_group_count", p.n_iter)))
        variants = max(1, int(getattr(p, "_arr_variants_per_ratio", 1)))
        expected_source_count = group_count * variants
        aligned_lengths = [
            len(getattr(p, "all_prompts", []) or []),
            len(getattr(p, "all_negative_prompts", []) or []),
            len(getattr(p, "all_seeds", []) or []),
            len(getattr(p, "all_subseeds", []) or []),
        ]
        source_count = min(min(aligned_lengths), expected_source_count)
        if source_count == 0:
            return
        if len(set(aligned_lengths + [expected_source_count])) > 1:
            print(
                f"{LOG} warning: prompt/seed array lengths differ or were changed "
                f"{aligned_lengths}, expected {expected_source_count}; using the "
                f"first {source_count} aligned item(s)"
            )

        ratio_count = self._effective_ratio_count(
            selected,
            getattr(p, "_arr_ratios_per_batch", 1),
            getattr(p, "_arr_selection_mode", RATIO_SELECTION_MODES[0]),
        )

        batch_ratio_plan: list[AspectRatio] = []
        expanded_prompts: list = []
        expanded_negative_prompts: list = []
        expanded_seeds: list = []
        expanded_subseeds: list = []
        expanded_hr_prompts: list | None = [] if self._has_aligned_list(p, "all_hr_prompts", source_count) else None
        expanded_hr_negative_prompts: list | None = [] if self._has_aligned_list(p, "all_hr_negative_prompts", source_count) else None

        usable_groups = math.ceil(source_count / variants)
        for group_index in range(usable_groups):
            batch_ratios = self._select_ratios_for_batch(
                selected,
                ratio_count,
                getattr(p, "_arr_selection_mode", RATIO_SELECTION_MODES[0]),
                group_index,
            )

            for ratio in batch_ratios:
                group_lo = group_index * variants
                group_hi = min(group_lo + variants, source_count)

                for chunk_lo in range(group_lo, group_hi, batch_size):
                    chunk_hi = min(chunk_lo + batch_size, group_hi)

                    batch_ratio_plan.append(ratio)
                    expanded_prompts.extend(p.all_prompts[chunk_lo:chunk_hi])
                    expanded_negative_prompts.extend(p.all_negative_prompts[chunk_lo:chunk_hi])
                    expanded_seeds.extend(p.all_seeds[chunk_lo:chunk_hi])
                    expanded_subseeds.extend(p.all_subseeds[chunk_lo:chunk_hi])

                    if expanded_hr_prompts is not None:
                        expanded_hr_prompts.extend(p.all_hr_prompts[chunk_lo:chunk_hi])
                    if expanded_hr_negative_prompts is not None:
                        expanded_hr_negative_prompts.extend(p.all_hr_negative_prompts[chunk_lo:chunk_hi])

        p.all_prompts = expanded_prompts
        p.all_negative_prompts = expanded_negative_prompts
        p.all_seeds = expanded_seeds
        p.all_subseeds = expanded_subseeds

        if expanded_hr_prompts is not None:
            p.all_hr_prompts = expanded_hr_prompts
        if expanded_hr_negative_prompts is not None:
            p.all_hr_negative_prompts = expanded_hr_negative_prompts

        p.n_iter = len(batch_ratio_plan)
        p._arr_plan = batch_ratio_plan
        p._arr_gallery_sort_keys = []

        if len({(r.antecedent, r.consequent) for r in batch_ratio_plan}) > 1:
            p.do_not_save_grid = True

        total = len(p.all_prompts)
        ratio_list = ", ".join(f"{r.antecedent}:{r.consequent}" for r in batch_ratio_plan)
        print(
            f"{LOG} {usable_groups} group(s) x {variants} variant(s) x "
            f"{ratio_count} ratio(s) = {total} image(s) | "
            f"{p.n_iter} GPU batch(es), GPU batch {p.batch_size}"
            f"\n{LOG} batch ratio plan: {ratio_list}"
        )

    def before_process_batch(self, p, enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery, **kwargs):
        """Apply this batch's resolution and rebuild the noise to match.

        The pipeline builds ``_shape``/``p.rng`` from ``p.width``/``p.height`` just
        *before* this hook fires (processing.py), so simply changing the dimensions
        is not enough — we must rebuild ``p.rng`` ourselves here.
        """
        if not getattr(p, "_arr_active", False):
            return

        batch_number = kwargs.get("batch_number", 0)
        plan_len = len(p._arr_plan)
        if plan_len == 0:
            return

        plan_index = batch_number
        if plan_index >= plan_len:
            if not p._arr_overflow_warned:
                p._arr_overflow_warned = True
                print(
                    f"{LOG} warning: batch count changed to {p.n_iter} after planning "
                    f"(expected {plan_len}) — cycling the resolution "
                    f"plan to cover the extra batches"
                )
            plan_index %= plan_len

        ratio = p._arr_plan[plan_index]
        p.width, p.height = calc_nearest_res_for_ratio(p._arr_base_w, ratio)

        # Record the chosen ratio in this batch's image metadata (PNG info / infotext).
        p.extra_generation_params["Aspect ratio"] = f"{ratio.antecedent}:{ratio.consequent}"

        if opts.data.get("arr_log_each_batch", True):
            print(
                f"{LOG} batch {batch_number + 1}/{p.n_iter} | "
                f"ratio {plan_index + 1}/{plan_len} ({ratio.antecedent}:{ratio.consequent}) "
                f"-> {p.width}x{p.height} | seeds {p.seeds}"
            )

        self._rebuild_rng(p)

        # init() computed hires targets once from the original size; recompute them
        # for this batch's base resolution.
        if getattr(p, "enable_hr", False) and hasattr(p, "calculate_target_resolution"):
            p.calculate_target_resolution()

    def postprocess_image_after_composite(self, p, pp, enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery):
        if not getattr(p, "_arr_active", False):
            return

        batch_index = getattr(p, "batch_index", 0)
        if batch_index >= len(getattr(p, "seeds", [])):
            return

        try:
            seed = int(p.seeds[batch_index])
        except (TypeError, ValueError):
            seed = 0

        p._arr_gallery_sort_keys.append(
            (seed, int(p.width), int(p.height), len(p._arr_gallery_sort_keys))
        )

    def postprocess(self, p, processed, enabled, ratios, variants_per_ratio, ratios_per_batch, selection_mode, sort_gallery):
        if not getattr(p, "_arr_active", False) or not getattr(p, "_arr_sort_gallery", True):
            return

        sort_keys = getattr(p, "_arr_gallery_sort_keys", [])
        first = getattr(processed, "index_of_first_image", 0)
        sortable_count = len(sort_keys)

        if sortable_count <= 1:
            return

        if len(processed.images) < first + sortable_count or len(processed.infotexts) < first + sortable_count:
            print(
                f"{LOG} warning: gallery sort skipped because the processed image "
                f"count did not match the generated image count"
            )
            return

        order = sorted(range(sortable_count), key=lambda i: sort_keys[i])

        def reorder_slice(values):
            values[first:first + sortable_count] = [
                values[first + i] for i in order
            ]

        reorder_slice(processed.images)
        reorder_slice(processed.infotexts)

        for attr in ("all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds"):
            values = getattr(processed, attr, None)
            if isinstance(values, list) and len(values) >= first + sortable_count:
                reorder_slice(values)

        if processed.infotexts:
            processed.info = processed.infotexts[0]
        if first == 0 and processed.all_seeds:
            processed.seed = int(processed.all_seeds[0])
        if first == 0 and processed.all_subseeds:
            processed.subseed = int(processed.all_subseeds[0])

    @staticmethod
    def _effective_ratio_count(selected: list[AspectRatio], requested: int, mode: str) -> int:
        requested = max(1, int(requested or 1))
        if mode == "Random with replacement":
            return requested
        return min(requested, len(selected))

    @staticmethod
    def _largest_divisor_at_most(value: int, maximum: int) -> int:
        value = max(1, int(value or 1))
        maximum = max(1, min(int(maximum or 1), value))

        for candidate in range(maximum, 0, -1):
            if value % candidate == 0:
                return candidate

        return 1

    @staticmethod
    def _has_aligned_list(p, attr: str, expected_len: int) -> bool:
        value = getattr(p, attr, None)
        return isinstance(value, list) and len(value) == expected_len

    @staticmethod
    def _select_ratios_for_batch(
        selected: list[AspectRatio],
        ratio_count: int,
        mode: str,
        batch_index: int,
    ) -> list[AspectRatio]:
        if mode == "Selected order":
            return selected[:ratio_count]

        if mode == "Sequential rotating":
            return [
                selected[(batch_index * ratio_count + offset) % len(selected)]
                for offset in range(ratio_count)
            ]

        if mode == "Random with replacement":
            return random.choices(selected, k=ratio_count)

        if ratio_count >= len(selected):
            ratios = selected[:]
            random.shuffle(ratios)
            return ratios

        return random.sample(selected, ratio_count)

    @staticmethod
    def _rebuild_rng(p):
        """Rebuild the noise RNG so the latent matches the new resolution.

        The pipeline already built ``p.rng`` for this batch (processing.py), so we
        reuse its shape and only swap the trailing spatial dims. This automatically
        preserves the latent channel count and any extra dimension used by Wan-based
        models (e.g. Anima), instead of reconstructing the shape from scratch.
        """
        opt_f = processing_module.opt_f  # runtime VAE downscale factor (usually 8)
        old_shape = tuple(p.rng.shape)
        new_shape = old_shape[:-2] + (p.height // opt_f, p.width // opt_f)
        p.rng = rng_module.ImageRNG(
            new_shape,
            p.seeds,
            subseeds=p.subseeds,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
        )


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
        "arr_log_each_batch",
        OptionInfo(
            True,
            "Log each batch's resolution to the console",
            section=section,
        ).info(
            "Prints a line per batch showing the chosen ratio, resolution and seeds. Disable to keep the console quiet on large runs (the run summary is always printed)."
        ),
    )


on_ui_settings(on_settings)


print(f"{LOG} loaded")
