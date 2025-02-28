import json

import datasets


class ImageChat(datasets.GeneratorBasedBuilder):
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
                    "file": "data/image_chat/train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file": "data/image_chat/valid.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file": "data/image_chat/test.json",
                },
            ),
        ]

    def _generate_examples(self, file):
        with open(file) as f:
            episode = json.load(f)
        i = 0
        for entry in episode:
            if len(entry['dialog']) < 2:
                continue

            context = f'[SEP]'.join([e[-1].lower() for e in entry['dialog'][:-1]])

            a1, b1 = context.split('[SEP]')
            conversation = f"[INST] [/INST] {a1.strip()} </s><s> [INST] {b1.strip()} [/INST]"
            response = entry['dialog'][-1][-1].lower()

            i += 1
            yield i, {
                "context": context,
                "conversation": conversation,
                "response": response,
            }
