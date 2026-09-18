# Third-party license mapping

The root `LICENSE` applies only to LombardTokenizer-specific contributions.
Third-party-origin modules retain their upstream terms.

## EnCodec-derived codec code

The codec building blocks adapted from the Meta/Facebook EnCodec codebase are
covered by the MIT license. The applicable text is in
[`ENCODEC-MIT.txt`](ENCODEC-MIT.txt), and the upstream source is
<https://github.com/facebookresearch/encodec>.

This mapping covers the EnCodec-derived portions of:

- `lombardtokenizer/codec/conv.py`
- `lombardtokenizer/codec/lstm.py`
- `lombardtokenizer/codec/norm.py`
- `lombardtokenizer/codec/quantization.py`
- `lombardtokenizer/codec/rvq.py`
- `lombardtokenizer/codec/seanet.py`

## vector-quantize-pytorch

`lombardtokenizer/codec/rvq.py` identifies the portions inspired by Phil
Wang's `vector-quantize-pytorch` implementation. The applicable MIT text is
in [`VECTOR-QUANTIZE-MIT.txt`](VECTOR-QUANTIZE-MIT.txt). Upstream:
<https://github.com/lucidrains/vector-quantize-pytorch>.

## Scientific references

SpeechTokenizer and AcademiCodec informed the scientific context of the
project, but no source code or checkpoint from either project is redistributed
here. They therefore do not require a license file in this repository.

## External mHuBERT weights

The repository does not include mHuBERT weights. The configured
`utter-project/mHuBERT-147` model is external and its model card identifies
CC-BY-NC-SA-4.0 terms:
<https://huggingface.co/utter-project/mHuBERT-147>.
