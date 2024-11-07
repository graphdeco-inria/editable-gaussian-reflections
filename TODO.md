
To implement:

- Make sure BRDF finetuning works with densification (bring back old checks for mlp)

- Save the BRDF LUT images, finetuned and direct
- Write down the convention for the BRDF LUT (in both uv format & pixel coordinates)

-------------------------

To compare:
- ground truth BRDF, from **g.t.** parameters
- ground truth BRDF, from **attached** parameters
- BRDF lut **without** fine-tuning, from **g.t.** parameters
- BRDF lut **without** fine-tuning, from **attached** parameters
- BRDF lut **with** fine-tuning, from **g.t.** parameters
- BRDF lut **with** fine-tuning, from **attached** parameters

---------------

- Add a flag to render bounce 1 
- Figure out what to do with antialiasing. How do I get multiple samples, where only the first one goes thru each pixel?

- Update the scene so theres no weird shaders.
- Double check the scene in the small study folder
- Upload new scenes & re-render
