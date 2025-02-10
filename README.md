# VIKDF

This is the code for [Distilling implicit multimodal knowledge into large language models for zero-resource dialogue generation](https://arxiv.org/abs/2405.10121).

## Requirements

* Python 3.11
* Pytorch 2.2.1
* CUDA 12.1

To install the Python dependencies, run:

```bash
pip install -r requirements.txt
```

To install nlg-eval, run:

```bash
git clone https://github.com/Maluuba/nlg-eval
cd nlg-eval
pip install -e .
```

To make the code work, some files need to be modified:
* `nlg-eval/requirements.txt`: change `gensim~=3.8.3` to `gensim>=3.8.3`
* `nlg-eval/nlgeval/word2vec/evaluate.py`: replace line 40 with the following line:

```python
return vectors[self.m.key_to_index[key]]
```

## Datasets
### Download MS COCO 2017
This example uses COCO dataset (2017) through a custom dataset script, which requires users to manually download the COCO dataset before training.
```bash
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```
### Download Reddit Conversation
Please download the Reddit data from [here](https://github.com/jokieleung/Maria).

### Download Image-Chat
The Image-Chat dataset can be accessed via [ParlAI](https://github.com/facebookresearch/ParlAI), with -t image_chat.

## Running Codes

Training:

```bash
bash scripts/run_train.sh
```

Inference:

```bash
bash scripts/run_inference.sh
```

## Reference

If you use any source code included in this repo in your work, please cite the following paper.

```
@article{ZHANG2025102985,
    title = {Distilling implicit multimodal knowledge into large language models for zero-resource dialogue generation},
    journal = {Information Fusion},
    volume = {118},
    pages = {102985},
    year = {2025},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2025.102985},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253525000582},
    author = {Bo Zhang and Hui Ma and Jian Ding and Jian Wang and Bo Xu and Hongfei Lin}
}
```
