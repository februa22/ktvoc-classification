# coding=utf-8
"""KT-VOC Classification Problem."""

import os
import random
from shutil import copyfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators.text_problems import VocabType
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class KtvocNoun587k(text_problems.Text2ClassProblem):
    """KT-VOC classification."""

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        """A single call to `generate_samples` generates for all `dataset_splits`.

        Set to True if you already have distinct subsets of data for each dataset
        split specified in `self.dataset_splits`. `self.generate_samples` will be
        called once for each split.

        Set to False if you have a unified dataset that you'd like to have split out
        into training and evaluation data automatically. `self.generate_samples`
        will be called only once and the data will be sharded across the dataset
        splits specified in `self.dataset_splits`.

        Returns:
            bool
        """
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate samples of text and label pairs.

        Each yielded dict will be a single example. The inputs should be raw text.
        The label should be an int in [0, self.num_classes).

        Args:
            data_dir: final data directory. Typically only used in this method to copy
                over user-supplied vocab files (for example, if vocab_type ==
                VocabType.TOKEN).
        tmp_dir: temporary directory that you can use for downloading and scratch.
        dataset_split: problem.DatasetSplit, which data split to generate samples
            for (for example, training and evaluation).

        Yields:
            {"inputs": text, "label": int}
        """
        # dataset_filename = 'kt.voc.noun.587k'
        dataset_filename = self.dataset_filename()
        data_path = os.path.join(tmp_dir, f'{dataset_filename}.pairs')
        return text2class_txt_iterator(data_path,
                                       class_strs=self.class_labels(data_dir))

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
            * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
                Must provide `self.approx_vocab_size`. Generates the vocabulary based on
                the training data. To limit the number of samples the vocab generation
                looks at, override `self.max_samples_for_vocab`. Recommended and
                default.
            * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
            * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
                vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
                will not be generated for you. The vocab file should be stored in
                `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
            VocabType constant
        """
        return VocabType.SUBWORD

    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**15  # ~32k

    @property
    def num_classes(self):
        """The number of classes."""
        return 135

    def class_labels(self, data_dir):
        """String representation of the classes."""
        class_labels_path = os.path.join(data_dir, self.class_labels_filename)
        class_labels = []
        with tf.gfile.Open(class_labels_path) as f:
            for line in f:
                class_labels.append(line.strip())
        return class_labels

    @property
    def class_labels_filename(self):
        return 'kt.voc.135.cls'

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        for i, sample in enumerate(
                self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
            yield sample["inputs"]
            if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
                break

    def create_class_labels(self, data_dir, tmp_dir):
        src = os.path.join(tmp_dir, self.class_labels_filename)
        dst = os.path.join(data_dir, self.class_labels_filename)
        copyfile(src, dst)

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        self.create_class_labels(data_dir, tmp_dir)
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        for sample in generator:
            inputs = encoder.encode(sample["inputs"])
            inputs.append(text_encoder.EOS_ID)
            label = sample["label"]
            yield {"inputs": inputs, "targets": [label]}

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

        return {
            "inputs": encoder,
            "targets": text_encoder.ClassLabelEncoder(self.class_labels(data_dir))
        }

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        source_vocab_size = self._encoders["inputs"].vocab_size
        p.input_modality = {
            "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
        }
        p.target_modality = (registry.Modalities.CLASS_LABEL, self.num_classes)

    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.FixedLenFeature([1], tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def eval_metrics(self):
        return [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5]


@registry.register_problem
class KtvocNoun72kShuffled(KtvocNoun587k):
    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**12  # ~4k

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        self.create_class_labels(data_dir, tmp_dir)
        generator = self.generate_shuffled_samples(
            data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        for sample in generator:
            inputs = encoder.encode(sample["inputs"])
            inputs.append(text_encoder.EOS_ID)
            label = sample["label"]
            yield {"inputs": inputs, "targets": [label]}

    def generate_shuffled_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        samples = list(generator)
        random.shuffle(samples)
        for sample in samples:
            yield sample


@registry.register_problem
class KtvocNoun587kShuffled(KtvocNoun72kShuffled):
    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**12  # ~4k


def txt_line_iterator(txt_path):
    """Iterate through lines of file."""
    with tf.gfile.Open(txt_path) as f:
        for line in f:
            yield line.strip()


def txt_tab_iterator(txt_path):
    for line in txt_line_iterator(txt_path):
        yield line.split('\t')


def text2class_txt_iterator(txt_path, class_strs=None):
    """Yield dicts for Text2ClassProblem.generate_samples from lines of files.

    Args:
        txt_path: path to txt file with a record per line, label and input
            are tab-separated.
        class_strs: list<str> of class label names. Must be in correct order (i.e.
            ["a", "b", "c"] means that "a" will get class ID 0, "b" ID 1, etc.).

    Yields:
        {"inputs": inputs, "label": label}
    """
    if class_strs:
        class_strs = dict([(s, i) for i, s in enumerate(class_strs)])
    for label, inputs in txt_tab_iterator(txt_path):
        label = label.strip()
        if class_strs:
            label = class_strs[label]
        else:
            label = int(label)
        yield {"inputs": inputs, "label": label}
