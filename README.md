# Aspect Ratio Randomizer

A Stable Diffusion WebUI script that enables random aspect ratio generation while maintaining consistent image area.

## Features

- Supports common aspect ratios (21:9, 16:9, 3:2, 4:3, 1:1, etc.)
- Supports custom aspect ratios via Settings menu
- SAutomatically adjusts dimensions (default 64px multiples) to maintain extension compatibility
- Maintains consistent image quality by preserving total pixel area

## Usage

1. Enable the "Aspect Ratio Randomizer" script in the `Scripts` section of the WebUI
1. Set your width to the desired base size.
    - Example: A width of `1024` will produce images within that resolution range. A ratio of 3:4 will produce a `896 x 1152` image.
1. Select your desired aspect ratios from the checkbox group
1. Set your batch count/size ([See Limitations](#limitations))
1. Run your generation as normal

## How?

The randomizer works through the following process:

1. Takes a base width and calculates dimensions that:
   - Match the chosen aspect ratio
   - Are divisible by 64 (default) pixels
       - Maintains compatibility with other extensions such as HiDiffusion

2. Area preservation is achieved by:
   - Using the square root of the base area multiplied by the ratio
   - This ensures the total pixel count stays relatively constant

## Limitations

- Cannot generate batches - it must run them sequentially
    - This is like setting `batch count`. The extension will use `batch count * batch size` as the final image count, so use either option to set the total number of generations.
- Incompatible with `img2img` because the resolution change will distort the output