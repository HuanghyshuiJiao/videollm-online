import os

from .stream import StreamMixIn


class DemoOverfit(StreamMixIn):
    evaluation_kwargs = {}

    def __init__(self, *, demo_overfit_feature_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.feature_dir = demo_overfit_feature_dir
        self.annos = [
            {
                "feature": "cooking.pt",
                "conversation": [
                    {"role": "user", "content": "Please narrate the cooking video in real time."},
                    {"role": "stream", "num_frames": 4, "learn": True},
                    {"role": "assistant", "content": "A person is preparing food at a kitchen counter.", "learn": True},
                    {"role": "stream", "num_frames": 4, "learn": True},
                    {"role": "assistant", "content": "They continue handling ingredients and kitchen tools.", "learn": True},
                    {"role": "user", "content": "What is happening in this clip?"},
                    {"role": "assistant", "content": "It shows a cooking or food-preparation scene.", "learn": True},
                ],
            },
            {
                "feature": "bicycle.pt",
                "conversation": [
                    {"role": "user", "content": "Please narrate the bicycle video in real time."},
                    {"role": "stream", "num_frames": 4, "learn": True},
                    {"role": "assistant", "content": "A bicycle is visible in the scene.", "learn": True},
                    {"role": "stream", "num_frames": 4, "learn": True},
                    {"role": "assistant", "content": "The clip focuses on a bicycle-related moment.", "learn": True},
                    {"role": "user", "content": "What object should I remember from this video?"},
                    {"role": "assistant", "content": "Remember the bicycle.", "learn": True},
                ],
            },
        ]

    def __getitem__(self, index):
        anno = self.annos[index % len(self.annos)]
        feature_path = os.path.join(self.feature_dir, anno["feature"])
        load_ranges = {feature_path: range(0, 8)}
        return *super().__getitem__(
            conversation=anno["conversation"],
            load_ranges=load_ranges,
        ), index, self.evaluation_kwargs

    def __len__(self):
        return len(self.annos)


def build_demo_overfit(**kwargs):
    return DemoOverfit(**kwargs)
