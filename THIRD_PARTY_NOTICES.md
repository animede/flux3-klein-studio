# Third-party models and licenses

This repository contains application source code. It does **not** distribute model
weights. The `models/` directory and generated images are excluded from Git.

The repository's [Apache License 2.0](LICENSE) applies only to the application code
authored for Klein Reference Studio. It does not grant rights to model weights,
model names, third-party software, or generated content. The controlling license is
always the license published by each provider.

## FLUX 3 Image Klein

- Inference model: [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
- Training/research checkpoint, not used by this application: [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)
- License: [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)
- Usage rules: [Black Forest Labs Usage Policy](https://bfl.ai/legal/usage-policy)

The FLUX 3 weights are not covered by this repository's Apache-2.0 license. The
FLUX Kommunity License permits non-commercial use and contains a commercial-output
exception for qualifying users with less than US$5 million in gross annualized
revenue, including affiliates, subject to its complete terms. It is not blanket
permission for commercial fine-tuning, model distribution, or every commercial
use. The license also contains use restrictions and requirements for derivatives
and redistribution.

The downloader therefore requires `--acknowledge-flux3-license`. This records only
the operator's acknowledgement; it does not replace acceptance of the gated model
terms on Hugging Face and does not modify the license. The application downloads
the distilled inference checkpoint only. It does not download the Base checkpoint
or the separate prompt upsampler.

## Python packages

Packages installed from `requirements.txt` remain under their respective licenses.
They are downloaded from their package providers and are not relicensed by this
repository. Use `pip show <package>` and the linked upstream project when reviewing
a specific dependency.

## Names and marks

FLUX and Black Forest Labs names and marks belong to their respective owners. This
project is an independent integration and is not affiliated with or endorsed by
Black Forest Labs or Hugging Face.

This notice is a practical summary, not a substitute for the license texts. If a
summary conflicts with a provider's terms, the provider's terms control.
