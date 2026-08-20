# Image-generation record

Mode: built-in OpenAI image generation with a local reference image.

Reference: `freedoom-demon-reference.png`, extracted from the BSD-3-Clause
Freedoom `freedoom2.wad`. The reference is supplied only for pose categories,
eight-view ordering, cell layout, camera height, and approximate sprite scale.
It must not be copied for creature identity, anatomy, silhouette, colors, or
surface design.

Prompt:

> Use case: stylized-concept
>
> Asset type: rotation-complete melee-enemy spritesheet for a Doom-compatible
> reinforcement-learning environment
>
> Input images: Image 1 is a BSD-licensed Freedoom layout reference used ONLY
> for the four-row grid, eight-view order, pose categories, camera height, and
> approximate per-cell scale. Do not copy its creature identity, worm anatomy,
> facial structure, fleshy texture, silhouette, colors, or distinctive design.
>
> Primary request: create a wholly original alternate melee enemy called the
> Verdigris Ram Hound as one exact 8-column by 4-row spritesheet. It is a squat
> four-legged clockwork hunting beast with a low wedge-shaped head, broad plated
> shoulders, short powerful legs, a compact barrel torso, and a small rigid tail.
> Its identity and proportions must remain identical in every occupied cell.
>
> Row 1: neutral locomotion/ready pose in eight clockwise views: front,
> front-right, right, back-right, back, back-left, left, front-left.
>
> Row 2: alternate stride and melee wind-up in the same eight-view order.
>
> Row 3: forward jaw-strike/bite pose in the same eight-view order. Keep the
> attack attached to the body; no projectile, particles, blood, or detached
> effects.
>
> Row 4: only the first three cells are occupied: front-facing pain recoil,
> collapsing body, and fully grounded corpse. The remaining five cells must be
> completely empty flat background.
>
> Style/medium: deliberately low-resolution 1990s FPS pixel-art sprites, hard
> pixel clusters, limited palette, no antialiasing, no painterly rendering, no
> 3D-render smoothness, suitable for nearest-neighbor reduction to 56 pixels
> tall.
>
> Scene/backdrop: perfectly flat solid #ff00ff chroma-key background covering
> the entire canvas and every empty cell. No shadows, gradients, texture,
> checkerboard, floor plane, reflections, grid lines, labels, or lighting
> variation.
>
> Composition/framing: exact 8 equal-width columns and 4 equal-height rows;
> one centered full-body sprite in every requested occupied cell; consistent
> baseline, scale, camera height, lighting, head size, torso size, and padding;
> generous separation; no overlap; every limb and tail remains inside its cell.
>
> Color palette: dark aged bronze, turquoise-green verdigris, charcoal joints,
> small warm amber eye slit, pale ceramic teeth. Do not use magenta, pink, red
> flesh, or the reference creature's palette anywhere in the character.
>
> Materials/textures: angular oxidized metal plates and compact mechanical
> joints. Original design only. No worm body, no serpentine body, no exposed
> organs, no tentacles, no horns, no recognizable copyrighted character.
>
> Constraints: crisp closed silhouettes; exactly four legs in every live pose;
> all body parts connected; consistent left/right anatomy under rotation; no
> text, logo, trademark, watermark, cast shadow, contact shadow, or extra object.
