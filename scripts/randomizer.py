import gradio as gr

from modules import scripts
from modules.ui_components import InputAccordion


ASPECT_RATIOS: dict = {
    '21:9': (1536, 640),
    '16:9': (1344, 768),
    '3:2': (1216, 832),
    '4:3': (1152, 896),
    '1:1': (1024, 1024),
    '2:3': (832, 1216),
    '3:4': (896, 1152),
    '9:16': (640, 1344),
    '9:21': (640, 1536),
}

class AspectRatioRandomizer(scripts.Script):
    sorting_priority = 1
    
    def title(self):
        return "AR Randomizer"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            aspect_ratio_checkboxes = []

            with gr.Row():
                col1, col2, col3 = gr.Column(), gr.Column(), gr.Column()
                columns = [col1, col2, col3]
                for i, (label, _) in enumerate(ASPECT_RATIOS.items()):
                    with columns[i % 3]:
                        chkbox = gr.Checkbox(label=label, value=False)
                        aspect_ratio_checkboxes.append(chkbox)

            return enabled, aspect_ratio_checkboxes
        
    def before_process(self, p, *args):
        x = super().before_process(p, *args)
        return x
    
    def before_process_batch(self, p, *args, **kwargs):
        return super().before_process_batch(p, *args, **kwargs)

