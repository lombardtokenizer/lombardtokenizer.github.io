# LombardTokenizer

LombardTokenizer is a neural speech codec for disentangling and controlling
vocal effort. It uses residual vector quantization to separate semantic
content, vocal effort, and residual acoustic information. The project was
presented at Interspeech 2025 in Rotterdam.

## Project page and audio samples

The [project page](https://lombardtokenizer.github.io/) presents the paper and
the accompanying audio demonstrations. It includes:

- synthesis and reconstruction examples;
- intra-speaker vocal-effort conversion;
- inter-speaker vocal-effort conversion;
- comparisons with the baselines used in the paper.

The samples used by the page are stored in `docs/assets/audio/`. Their status
is documented separately in [`docs/assets/audio/NOTICE`](docs/assets/audio/NOTICE)
and is not inferred from the code license.

## Paper

**LombardTokenizer: Disentanglement and Control of Vocal Effort in a Neural
Speech Codec**
Maxime Jacquelin, Maëva Garnier, Laurent Girin, Rémy Vincent, and Olivier
Perrotin
Interspeech 2025, Rotterdam
[Paper and DOI](https://doi.org/10.21437/Interspeech.2025-1639)

## Repository contents

~~~text
lombardtokenizer/       Python package
configs/                training configurations
scripts/                training and inference commands
docs/                   project page and paper audio demonstrations
licenses/               third-party license texts and mappings
~~~

## Installation

From the repository root:

~~~bash
python -m pip install -e .
~~~

Semantic feature extraction additionally requires Transformers:

~~~bash
python -m pip install -e ".[semantic]"
~~~

The package requires Python 3.8 or newer, PyTorch, torchaudio, NumPy,
einops, and PyYAML. The complete dependency declaration is in
[`pyproject.toml`](pyproject.toml); `requirements.txt` provides the editable
installation with the semantic extra.

## Architecture

The released LT2 recipe uses eight residual vector-quantization stages:

~~~text
VQ1       -> semantic representation
VQ2       -> vocal effort
VQ3-VQ8   -> residual acoustic information
~~~

VQ1 is distilled from mHuBERT features and VQ2 is conditioned on an E2
vocal-effort encoder. The external `utter-project/mHuBERT-147` model is used
for semantic features; its checkpoint is downloaded separately and is not
redistributed here. The semantic hidden layer must be selected explicitly for
each experiment.

## Data preparation

### mHuBERT features

The semantic extraction manifest uses two tab-separated fields:

~~~text
audio_path<TAB>semantic_feature_path
data/audio/spk1_001.wav<TAB>data/mhubert/spk1_001.npy
~~~

Extract features with a layer selected for the experiment:

~~~bash
python scripts/extract_semantic_features.py --model utter-project/mHuBERT-147 --audio-dir data/audio --output-dir data/mhubert --layer 8
~~~

The value `8` is an example invocation, not a paper-verified default. Replace
it with the hidden layer selected for the experiment.

### Vocal-effort encoder

The E2 manifest uses three tab-separated fields:

~~~text
audio_path<TAB>speaker_id<TAB>intensity
data/audio/spk1_001.wav<TAB>spk1<TAB>72.4
~~~

Prepare the train and validation manifests referenced by
`configs/vocal_effort_encoder.yaml` before training.

## Train the vocal-effort encoder

~~~bash
python scripts/train_vocal_effort_encoder.py --config configs/vocal_effort_encoder.yaml
~~~

The E2 checkpoint stores the speaker-wise intensity normalization statistics
and the complete encoder configuration.

## Train LombardTokenizer

Set `semantic.layer` and `vocal_effort.checkpoint` in
`configs/lt2.yaml`, then run:

~~~bash
python scripts/train.py --config configs/lt2.yaml
~~~

The codec checkpoint embeds the model configuration. `last.pt` stores the
latest training state and `best.pt` stores the best validation state when a
validation manifest is configured.

## Resume training

Resuming restores the model, optimizer, scheduler, epoch, step, best
validation loss, and relevant training statistics:

~~~bash
python scripts/train.py --config configs/lt2.yaml --resume path/to/last.pt
~~~

Replace `path/to/last.pt` with the checkpoint to resume.

## Fine-tuning

Fine-tuning starts from model weights with a fresh optimizer and scheduler:

~~~bash
python scripts/finetune.py --config configs/lt2_finetune.yaml --checkpoint path/to/pretraining.pt
~~~

Replace `path/to/pretraining.pt` with the pretraining checkpoint. To continue
a complete fine-tuning state instead, pass `--resume` rather than
`--checkpoint`.

## Reconstruction and vocal-effort conversion

Reconstruct a waveform with:

~~~bash
python scripts/reconstruct.py --checkpoint path/to/model.pt --input input.wav --output reconstruction.wav
~~~

Convert vocal effort by swapping VQ2 codes between a source and a reference:

~~~bash
python scripts/convert.py --checkpoint path/to/model.pt --source source.wav --reference reference.wav --output converted.wav
~~~

Replace the example paths with actual local files.

The Python API is also available:

~~~python
from lombardtokenizer import LombardTokenizer

model = LombardTokenizer.load_from_checkpoint("last.pt")
reconstruction = model.reconstruct(audio)
converted = model.convert_vocal_effort(source_audio, reference_audio)
~~~

VQ2 conversion requires source and reference codes with equal batch and time
dimensions; temporal interpolation is not implicit.

## Datasets

### AVID

AVID provides calibrated speech recordings in four vocal-effort categories and
is used for the demonstrations and fine-tuning described by the paper. See the
[AVID publication](https://www.sciencedirect.com/science/article/pii/S0167639324000116)
and its [Zenodo distribution](https://zenodo.org/records/7948300) for dataset
terms.

### FLombard

FLombard is the French Lombard-speech dataset used for evaluation. The dataset
record is available at
<https://doi.org/10.5281/zenodo.17340497>.

## Limitations

- the mHuBERT hidden layer must be selected for each experiment;
- the E2 frontend is an implementation choice;
- VQ2 conversion requires equal source and reference code lengths;
- paper-table reproduction and WER, EER, JS-F0, and JS-Slope evaluation are
  outside this package.

## Citation

~~~bibtex
@inproceedings{jacquelin25_interspeech,
  title = {{LombardTokenizer: Disentanglement and Control of Vocal Effort in a Neural Speech Codec}},
  author = {Maxime Jacquelin and Maëva Garnier and Laurent Girin and Rémy Vincent and Olivier Perrotin},
  year = {2025},
  booktitle = {{Interspeech 2025}},
  pages = {5778--5782},
  doi = {10.21437/Interspeech.2025-1639},
  issn = {2958-1796}
}
~~~

## Acknowledgements

The codec contains source-derived components from EnCodec and
vector-quantize-pytorch; their license texts and file mappings are documented
in [`NOTICE`](NOTICE) and [`licenses/README.md`](licenses/README.md).
SpeechTokenizer and AcademiCodec are cited as scientific and architectural
references. FreeVC and other comparison systems may appear on the project page
as paper baselines; they are not distributed by this repository.

## License

The root [`LICENSE`](LICENSE) applies to LombardTokenizer-specific code. The
third-party portions identified in [`NOTICE`](NOTICE) retain their upstream
licenses, whose texts are in [`licenses/`](licenses/). The external mHuBERT
checkpoint is not included and has independent model terms. Audio samples have
separate dataset and rights conditions documented in
[`docs/assets/audio/NOTICE`](docs/assets/audio/NOTICE).

The Python package configuration does not include `docs/assets/audio/` in its
wheel or source package data.
