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
        # Always-on accordion (like ControlNet/FreeU), coexists with other scripts.
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(
            False, label=self.title(), elem_id="arr-enabled"
        ) as enabled:
            if is_img2img:
                gr.Markdown(
                    "Aspect Ratio Randomizer only affects **txt2img** — changing the "
                    "resolution mid-run would distort an img2img init image, so it does "
                    "nothing here."
                )
                # Hidden placeholders so the returned arg count stays stable.
                ratios = gr.CheckboxGroup(
                    choices=list(ASPECT_RATIOS.keys()), visible=False
                )
                clamp_to_resolutions = gr.Checkbox(value=False, visible=False)
                match_seeds = gr.Checkbox(value=True, visible=False)
                variants_per_resolution = gr.Slider(
                    minimum=1, maximum=100, step=1, value=1, visible=False
                )
                return [enabled, ratios, clamp_to_resolutions, match_seeds, variants_per_resolution]

            with gr.Row():
                ratios = gr.CheckboxGroup(
                    label="Aspect Ratios",
                    choices=list(ASPECT_RATIOS.keys()),
                    info="Select the aspect ratios you want to randomize between. Order is: Wide - Square - Tall",
                )

            gr.Markdown(
                "**Batch count** = number of aspect ratios; **batch size** = max images "
                "generated concurrently (VRAM). The slider below sets how many variants to "
                "render per resolution (it can exceed the batch size limit of 8). "
                "Total images = batch count x variants per resolution."
            )

            variants_per_resolution = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=1,
                label="Variants per resolution",
                info="How many images to render at each resolution. Generated in GPU batches of 'batch size'. Can go higher than the batch size limit.",
            )

            match_seeds = gr.Checkbox(
                value=True,
                label="Use the same seed across all resolutions",
                info="Renders the same seed(s) at every aspect ratio so you can compare them directly. Uncheck to let seeds advance normally per batch.",
            )

            clamp_to_resolutions = gr.Checkbox(
                value=False,
                label="Limit batch count to number of selected resolutions",
                info="When enabled, the batch count is reduced so each selected ratio is used at most once (a message is logged).",
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

        return [enabled, ratios, clamp_to_resolutions, match_seeds, variants_per_resolution]

    def before_process(self, p, enabled, ratios, clamp_to_resolutions, match_seeds, variants_per_resolution):
        """Reconfigure the batch loop into resolution / variant / concurrency groups.

        Interpretation of the controls:
          - batch count (p.n_iter)    -> number of resolutions (R)
          - batch size (p.batch_size) -> max images generated concurrently (the GPU batch, L)
          - variants_per_resolution   -> how many variants to render per resolution (V)

        We rewrite the real ``p.n_iter`` / ``p.batch_size`` accordingly. This runs
        before ``setup_prompts``/``all_seeds`` are built (processing.py), so the
        rewrite correctly resizes the prompt/seed arrays and the job/step counters.
        Each GPU batch (``ceil(V / L)`` per resolution) stays at a single resolution.
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

        resolutions = p.n_iter           # batch count -> number of resolutions (R)
        variants = max(1, int(variants_per_resolution or 1))       # slider -> variants (V)
        concurrency = max(1, min(p.batch_size, variants))          # batch size -> GPU batch (L)

        # Optionally cap the number of resolutions so each selected ratio is used once.
        if clamp_to_resolutions and resolutions > len(selected):
            print(
                f"{LOG} limiting resolutions {resolutions} -> {len(selected)} "
                f"(one per selected ratio)"
            )
            resolutions = len(selected)

        # GPU batches needed to cover V variants at L-at-a-time (rounds up).
        chunks_per_res = math.ceil(variants / concurrency)

        # One ratio per resolution. Reshuffle each full pass so ratios don't repeat
        # until the whole selection has been used.
        plan: list[AspectRatio] = []
        pool: list[AspectRatio] = []
        for _ in range(resolutions):
            if not pool:
                pool = selected[:]
                random.shuffle(pool)
            plan.append(pool.pop())

        # Rewrite the real Forge batch params.
        p.batch_size = concurrency
        p.n_iter = resolutions * chunks_per_res

        p._arr_active = True
        p._arr_base_w = p.width
        p._arr_plan = plan
        p._arr_chunks_per_res = chunks_per_res
        # When matching seeds, each resolution's variant seeds (keyed by chunk index)
        # are captured during the first resolution and reused for the rest.
        p._arr_seed_bank: dict[int, list] = {}
        p._arr_subseed_bank: dict[int, list] = {}

        total = p.n_iter * p.batch_size
        ratio_list = ", ".join(f"{r.antecedent}:{r.consequent}" for r in plan)
        print(
            f"{LOG} {resolutions} resolution(s) x {chunks_per_res * concurrency} "
            f"variant(s) = {total} image(s) | GPU batch {concurrency}, {p.n_iter} batches"
            f"\n{LOG} ratios: {ratio_list}"
        )

        # Mixed resolutions make the output grid look broken.
        if len({(r.antecedent, r.consequent) for r in plan}) > 1:
            p.do_not_save_grid = True

    def before_process_batch(self, p, enabled, ratios, clamp_to_resolutions, match_seeds, variants_per_resolution, **kwargs):
        """Apply this batch's resolution and rebuild the noise to match.

        The pipeline builds ``_shape``/``p.rng`` from ``p.width``/``p.height`` just
        *before* this hook fires (processing.py), so simply changing the dimensions
        is not enough — we must rebuild ``p.rng`` ourselves here.
        """
        if not getattr(p, "_arr_active", False):
            return

        batch_number = kwargs.get("batch_number", 0)
        chunks_per_res = p._arr_chunks_per_res
        res_index = batch_number // chunks_per_res
        chunk_index = batch_number % chunks_per_res

        # Reuse each resolution's variant seeds (by chunk) so the same seeds render at
        # every aspect ratio. Must happen before the RNG rebuild (it reads p.seeds).
        if match_seeds:
            if res_index == 0:
                p._arr_seed_bank[chunk_index] = list(p.seeds)
                p._arr_subseed_bank[chunk_index] = list(p.subseeds)
            else:
                p.seeds = list(p._arr_seed_bank[chunk_index])
                p.subseeds = list(p._arr_subseed_bank[chunk_index])
                bs = p.batch_size
                lo = batch_number * bs
                hi = lo + bs
                p.all_seeds[lo:hi] = p.seeds
                p.all_subseeds[lo:hi] = p.subseeds

        ratio = p._arr_plan[res_index]
        p.width, p.height = calc_nearest_res_for_ratio(p._arr_base_w, ratio)

        # Record the chosen ratio in this batch's image metadata (PNG info / infotext).
        p.extra_generation_params["Aspect ratio"] = f"{ratio.antecedent}:{ratio.consequent}"

        if opts.data.get("arr_log_each_batch", True):
            resolutions = len(p._arr_plan)
            print(
                f"{LOG} batch {batch_number + 1}/{p.n_iter} | "
                f"resolution {res_index + 1}/{resolutions} ({ratio.antecedent}:{ratio.consequent}) "
                f"-> {p.width}x{p.height} | seeds {p.seeds}"
            )

        self._rebuild_rng(p)

        # init() computed hires targets once from the original size; recompute them
        # for this batch's base resolution.
        if getattr(p, "enable_hr", False) and hasattr(p, "calculate_target_resolution"):
            p.calculate_target_resolution()

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
