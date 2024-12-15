from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts
import gradio as gr

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
