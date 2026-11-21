# Publishing to Hugging Face (after a full train + evaluate + export)

Replace `YOUR_HF_USERNAME` with your Hugging Face username. Create a token with **write** access at
https://huggingface.co/settings/tokens

```
pip install -U huggingface_hub
hf auth login
```
(older versions: `huggingface-cli login`)

## 1. Model repo (weights + model card)
```
hf repo create pcam-tumor-patch-classifier --repo-type model
hf upload YOUR_HF_USERNAME/pcam-tumor-patch-classifier release . --repo-type model
```
`release/README.md` becomes the model card.

## 2. Demo Space
```
hf repo create pcam-tumor-patch-demo --repo-type space --space_sdk gradio
hf upload YOUR_HF_USERNAME/pcam-tumor-patch-demo space . --repo-type space
```
The Space builds by itself in a few minutes. Open `https://huggingface.co/spaces/YOUR_HF_USERNAME/pcam-tumor-patch-demo`.

## 3. Test it
- Upload one of the example patches in the Space, check that the disclaimer shows.
- `python predict.py some_patch.png --weights release/model.safetensors`

Tip: create the repos as **private** first (add `--private`), check everything, then switch to public in Settings.
