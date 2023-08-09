from utils import *
import sys

NUM_TRAIN = 40000
NUM_EVAL = 1000

set_seed(SEED)
logging.set_verbosity(logging.ERROR)
all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(".", use_token=True)
programs = json.load(open("seed_programs.json"))
program = (sys.argv[1], programs[sys.argv[1]])

for C in sys.argv[2:]:
    print(f"generating {C} ctf dataset of {program[0]} ...")
    preload_train_dataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            NUM_TRAIN,
            C,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )

    preload_validation_dataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            NUM_EVAL,
            C,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )

    preload_test_dataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            NUM_EVAL,
            C,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )
    preload_train_dataset.to_json(f"./preload_datasets/train_{program[0]}_{C}.json")
    preload_validation_dataset.to_json(f"./preload_datasets/validation_{program[0]}_{C}.json")
    preload_test_dataset.to_json(f"./preload_datasets/test_{program[0]}_{C}.json")
    