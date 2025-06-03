from datasets import load_dataset

"""
prepare flores-200 dataset with dataset library of python.

following code does not work in colab, I downloaded dataset by hand.

from: https://huggingface.co/datasets/openlanguagedata/flores_plus/tree/main/devtest
"""

import huggingface_hub

huggingface_hub.login()

# Clear cache and force re-download
# ds = load_dataset("facebook/flores", "all", download_mode="force_redownload")

huggingface_hub.snapshot_download(
    repo_id="openlanguagedata/flores_plus",
    local_dir="./datasets/"
)

# load dev and devtests splits for all languages
ds_full = load_dataset("openlanguagedata/flores_plus")
# load only the dev split for all languages
ds_dev = load_dataset("openlanguagedata/flores_plus", split="dev")
# load dev and devtests splits for French only
ds_fra = load_dataset("openlanguagedata/flores_plus", "fra_Latn")
# load dev split for French only
ds_fra_dev = load_dataset("openlanguagedata/flores_plus", "fra_Latn", split="dev")