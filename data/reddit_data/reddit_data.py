import json

import datasets


class RedditData(datasets.GeneratorBasedBuilder):
    """Reddit dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "conversation": datasets.Value("string"),
                    "response": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file": "data/reddit_data/train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file": "data/reddit_data/valid.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file": "data/reddit_data/test.json",
                },
            ),
        ]

    def _generate_examples(self, file):
        with open(file) as f:
            data = json.load(f)
        c = 0
        for episode in data:
            context = f'[SEP]'.join(episode['context'])
            response = episode['response']

            if len(episode['context']) == 3:
                a1, a2, a3 = episode['context']
                conversation = f'[INST] {a1.strip()} [/INST] {a2.strip()} </s><s> [INST] {a3.strip()} [/INST]'
            elif len(episode['context']) == 2:
                a1, a2 = episode['context']
                conversation = f'[INST] [/INST] {a1.strip()} </s><s> [INST] {a2.strip()} [/INST]'
            else:
                continue

            c += 1
            yield c, {
                "context": context,
                "conversation": conversation,
                "response": response,
            }