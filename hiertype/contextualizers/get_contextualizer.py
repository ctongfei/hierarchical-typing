from hiertype.contextualizers.contextualizer import Contextualizer
from hiertype.contextualizers.hugging_face_contextualizer import BERTContextualizer, XLMRobertaContextualizer
from hiertype.contextualizers.elmo_contextualizer import ELMoContextualizer


def get_contextualizer(
        model_name: str,
        device: str,
        tokenizer_only: bool = False
) -> Contextualizer:
    """
    Returns a contextualizer by pre-trained model name.

    :param model_name: Model identifier.
    :param device:
    :param tokenizer_only: if True, only loads tokenizer (not actual model)
    """

    if model_name.startswith("bert"):
        return BERTContextualizer.from_model(model_name, device, tokenizer_only=tokenizer_only)

    elif model_name.startswith("xlm-roberta"):
        return XLMRobertaContextualizer.from_model(model_name, device, tokenizer_only=tokenizer_only)

    elif model_name.startswith("elmo"):
        elmo_path_prefix = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo"
        elmo_path_id = {
            "elmo-small": "2x1024_128_2048cnn_1xhighway",
            "elmo-medium": "2x2048_256_2048cnn_1xhighway",
            "elmo-original": "2x4096_512_2048cnn_2xhighway",
            "elmo-original-5.5B": "2x4096_512_2048cnn_2xhighway_5.5B"
        }[model_name]
        elmo_path = f"{elmo_path_prefix}/{elmo_path_id}/elmo_{elmo_path_id}"
        elmo_weights_path = f"{elmo_path}_weights.hdf5"
        elmo_options_path = f"{elmo_path}_options.json"
        return ELMoContextualizer.from_model(
            elmo_weights_path, elmo_options_path, device, tokenizer_only=tokenizer_only
        )

    elif model_name.startswith("glove"):
        glove_path = {
            "glove-6B-300d": "http://nlp.stanford.edu/data/glove.6B.zip",
            "glove-42B-300d": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
            "glove-840B-300d": "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        }
        # TODO: read GloVe files
        raise NotImplementedError
