from datasets import load_dataset

def verbalize_sst2(sample):
    candidate = sample["label"]
    verbalizer = {0: "terrible", 1: "great"}
    text = sample["sentence"].strip()
    return f"{text} It was {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_rte(sample):
    verbalizer = {0: "Yes", 1: "No"}
    candidate = sample["label"]
    premise = sample["premise"]
    hypothesis = sample["hypothesis"]
    return f"{premise}\nDoes this mean that \"{hypothesis}\" is true? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_boolq(sample):
    candidate = "Yes" if sample["answer"] else "No"
    passage = sample["passage"]
    question = sample["question"].strip()
    if not question.endswith("?"):
        question += "?"
    question = question[0].upper() + question[1:]
    return f"{passage} {question} {candidate}", candidate

def verbalize_wsc(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    text = sample['text']
    span1 = sample['span1_text']
    span2 = sample['span2_text']
    return f"{text}\nIn the previous sentence, does the pronoun \"{span2.lower()}\" refer to? {span1}? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_wic(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    sent1 = sample["sentence1"]
    sent2 = sample["sentence2"]
    word = sample["word"]
    return f"Does the word \"{word}\" have the same meaning in these two sentences? Yes, No?\n{sent1}\n{sent2}\n {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_multirc(sample):
    candidate = "Yes" if sample["label"] == 1 else "No"
    paragraph = sample["paragraph"]
    question = sample["question"]
    answer = sample["answer"]
    return f"{paragraph}\nQuestion: {question}\nI found this answer \"{answer}\". Is that correct? Yes or No? {candidate}", candidate

def verbalize_copa(sample, mode="train"):
    def capitalize(c):
        words = c.split(" ")
        if words[0] != "I":
            words[0] = words[0].lower()
        return " ".join(words)
    candidate = sample["label"]
    premise = sample["premise"].rstrip()
    if premise.endswith("."):
        premise = premise[:-1]
    conjunction = " so " if sample["question"] == "effect" else " because "
    correct_choice = sample[f"choice{candidate + 1}"]
    correct_choice = capitalize(correct_choice)
    correct_sentence = f"{premise}{conjunction}{correct_choice}"
    
    # For evaluation, also return the wrong choice and sentence
    wrong_choice = sample[f"choice{2 - candidate}"]
    wrong_choice = capitalize(wrong_choice)
    wrong_sentence = f"{premise}{conjunction}{wrong_choice}"
    
    return correct_sentence, correct_choice, wrong_sentence, wrong_choice

def verbalize_squad(sample):
    question = sample['question'].strip()
    title = sample['title']
    context = sample['context']
    answer = sample['answers']['text'][0]
    return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}", answer

def verbalize_drop(sample):
    question = sample['question'].strip()
    passage = sample['passage']
    answer = sample['answers_spans']['spans'][0]
    return f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}", answer

def verbalize_winogrande(sample):
    context, target = sample["sentence"].split("_")
    verbalizer = {"1": sample['option1'] + target, "2": sample['option2'] + target}
    corect_candidate = sample["answer"] 
    wrong_candidate = "1" if corect_candidate == "2" else "2"
    return f"{context}{verbalizer[corect_candidate]}", verbalizer[corect_candidate], f"{context}{verbalizer[wrong_candidate]}", verbalizer[wrong_candidate]

def verbalize_mrpc(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    sentence1 = sample["sentence1"]
    sentence2 = sample["sentence2"]
    # return f"Can I replace the sentence:\n{sentence1}\nwith the sentence:\n{sentence2}\nand have it mean the same thing? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]
    # return f"Does the sentence\n{sentence1}\nparaphrase (that is, mean the same thing as) this sentence?\n{sentence2}\n Yes or No? {verbalizer[candidate]}", verbalizer[candidate]
    return f"Do the following two sentences mean the same thing? Yes or No?\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n {verbalizer[candidate]}", verbalizer[candidate]


def verbalize_qqp(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    question1 = sample["question1"]
    question2 = sample["question2"]
    # return f"Question 1: {question1}\nQuestion 2: {question2}\nDo these two questions convey the same meaning? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]
    return f"Are these two questions asking the same thing? Yes or No?\nQuestion 1: {question1}\nQuestion 2: {question2}\n {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_qnli(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    question = sample["question"]
    sentence = sample["sentence"]
    # return f"{sentence}\nDoes that sentence have all you need to answer the question {question}? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]
    return f"Does this sentence answer the question? Yes or No?\nQuestion: {question}\nSentence: {sentence}\n {verbalizer[candidate]}", verbalizer[candidate]

def verbalize_wnli(sample):
    verbalizer = {0: "No", 1: "Yes"}
    candidate = sample["label"]
    sentence1 = sample["sentence1"]
    sentence2 = sample["sentence2"]
    # return f"Assume that the following sentence is true:\n{sentence1}\nDoes this mean that {sentence2}? Yes or No? {verbalizer[candidate]}", verbalizer[candidate]
    return f"Given the first sentence, is the second sentence true? Yes or No?\nSentence 1: {sentence1}\nSentence 2: {sentence2}\n {verbalizer[candidate]}", verbalizer[candidate]

def load_data(args):
    if args.task_name == "sst2":
        raw_datasets = load_dataset("glue", "sst2")
        verbalize_fn = verbalize_sst2
    elif args.task_name == "rte":
        raw_datasets = load_dataset("super_glue", "rte")
        verbalize_fn = verbalize_rte
    elif args.task_name == "boolq":
        raw_datasets = load_dataset("boolq")
        verbalize_fn = verbalize_boolq
    elif args.task_name == "wsc":
        raw_datasets = load_dataset("super_glue", "wsc.fixed")
        verbalize_fn = verbalize_wsc
    elif args.task_name == "wic":
        raw_datasets = load_dataset("super_glue", "wic")
        verbalize_fn = verbalize_wic
    elif args.task_name == "multirc":
        raw_datasets = load_dataset("super_glue", "multirc")
        verbalize_fn = verbalize_multirc    
    elif args.task_name == "copa":
        raw_datasets = load_dataset("super_glue", "copa")
        verbalize_fn = verbalize_copa
    elif args.task_name == "squad":
        raw_datasets = load_dataset("squad")
        verbalize_fn = verbalize_squad
    elif args.task_name == "drop":
        raw_datasets = load_dataset("drop")
        verbalize_fn = verbalize_drop
    elif args.task_name == "winogrande":
        raw_datasets = load_dataset("winogrande", "winogrande_m")
        verbalize_fn = verbalize_winogrande
    elif args.task_name == "mrpc":
        raw_datasets = load_dataset("glue", "mrpc")
        verbalize_fn = verbalize_mrpc
    elif args.task_name == "qqp":
        raw_datasets = load_dataset("glue", "qqp")
        verbalize_fn = verbalize_qqp
    elif args.task_name == "qnli":
        raw_datasets = load_dataset("glue", "qnli")
        verbalize_fn = verbalize_qnli
    elif args.task_name == "wnli":
        raw_datasets = load_dataset("glue", "wnli")
        verbalize_fn = verbalize_wnli
    else:
        raise ValueError(f"Task {args.task_name} not supported.")

    if "test" in raw_datasets:
        del raw_datasets["test"]

    if len(raw_datasets["train"]) >= 1000:
        raw_datasets['train'] = raw_datasets['train'].select(range(1000))
    if len(raw_datasets["validation"]) >= 1000:
        raw_datasets['validation'] = raw_datasets['validation'].select(range(1000))

    raw_datasets["train"] = raw_datasets["train"].map(lambda x: {"sentence": verbalize_fn(x)}, load_from_cache_file=False)
    raw_datasets["validation"] = raw_datasets["validation"].map(lambda x: {"sentence": verbalize_fn(x)}, load_from_cache_file=False)

    return raw_datasets

def encode_data(args, tokenizer, raw_datasets):

    cls_idx = []
    def preprocess_function(examples, is_eval=False):
        sentences_with_candidates = examples["sentence"]
        
        labels = []
        input_ids_list = []
        for sentence_with_candidate in sentences_with_candidates:
            if len(sentence_with_candidate) == 2:
                # This is a sentence-candidate pair (classification-like input)
                sentence, candidate = sentence_with_candidate
                if "Llama" in args.model_name_or_path and "Llama-3" not in args.model_name_or_path: # Llama-3 has a different tokenization
                    pass
                elif "opt" in args.model_name_or_path or "gemma" in args.model_name_or_path or "Llama-3" in args.model_name_or_path:
                    candidate = " " + candidate
                else:
                    raise ValueError(f"Unknown model: {args.model_name_or_path}")
                
                # Tokenize the full sentence and the candidate/choice separately
                tokenized_example = tokenizer(sentence, max_length=args.max_length, truncation=True)
                tokenized_candidate = tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if not tokenized_candidate[0] in cls_idx:
                    cls_idx.append(tokenized_candidate[0])
                # print(f"Sentence: {sentence}, Candidate: {candidate}")
                # print(f"Tokenized sentence: {tokenized_example}")
                # print(f"Tokenized candidate: {tokenized_candidate}")
                
                input_ids = tokenized_example["input_ids"]
                label = [-100] * len(input_ids)
                
                # Label the last `len(tokenized_candidate)` tokens
                label[-len(tokenized_candidate):] = input_ids[-len(tokenized_candidate):]
                
                labels.append(label)
                input_ids_list.append(input_ids)

                if args.task_name != "copa" and args.task_name != "squad" and args.task_name != "drop":
                    assert input_ids[-len(tokenized_candidate):] == tokenized_candidate, f"Tokenized candidate does not match the last tokens of the input_ids: {input_ids[-len(tokenized_candidate):]} vs {tokenized_candidate}"
                    assert len(tokenized_candidate) == 1, f"Tokenized candidate is not a single token: {tokenized_candidate}"
            
            elif len(sentence_with_candidate) == 4:
                # This is a correct and wrong sentence-choice pair (multiple choice-like input)
                correct_sentence, correct_choice, wrong_sentence, wrong_choice = sentence_with_candidate
                
                # Tokenize the correct sentence and choice
                tokenized_correct_example = tokenizer(correct_sentence, max_length=args.max_length, truncation=True)
                tokenized_correct_choice = tokenizer(correct_choice, add_special_tokens=False)["input_ids"]

                input_ids = tokenized_correct_example["input_ids"]
                label = [-100] * len(input_ids)
                label[-len(tokenized_correct_choice):] = input_ids[-len(tokenized_correct_choice):]
                
                input_ids_list.append(input_ids)
                labels.append(label)
                
                if is_eval:
                    # Tokenize the wrong sentence and choice for evaluation
                    tokenized_wrong_example = tokenizer(wrong_sentence, max_length=args.max_length, truncation=True)
                    tokenized_wrong_choice = tokenizer(wrong_choice, add_special_tokens=False)["input_ids"]

                    input_ids = tokenized_wrong_example["input_ids"]
                    label = [-100] * len(input_ids)
                    label[-len(tokenized_wrong_choice):] = input_ids[-len(tokenized_wrong_choice):]

                    input_ids_list.append(input_ids)
                    labels.append(label)
        
        return {"input_ids": input_ids_list, "labels": labels}

    processed_datasets = {
        "train": raw_datasets["train"].map(lambda x: preprocess_function(x), batched=True, load_from_cache_file=False, remove_columns=raw_datasets["train"].column_names),
        "validation": raw_datasets["validation"].map(lambda x: preprocess_function(x, True), batched=True, load_from_cache_file=False, remove_columns=raw_datasets["validation"].column_names)
    }

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    
    
    return train_dataset, eval_dataset, cls_idx


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b", help="Path to pretrained model.")
    parser.add_argument("--task_name", type=str, default="copa", help="The name of the GLUE task to train on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum input sequence length after tokenization.")
    args = parser.parse_args()

    raw_datasets = load_data(args)

    print(len(raw_datasets['train']))
    print(len(raw_datasets['validation']))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side="left", truncation_side="left")

    train_dataset, eval_dataset, _ = encode_data(args, tokenizer, raw_datasets)

    print(train_dataset[0])
    print(eval_dataset[0])

    for i in range(10):
        # print(tokenizer.decode(train_dataset[i]["input_ids"]))
        # print(tokenizer.decode([x for x in train_dataset[i]["labels"] if x != -100]))

        print(tokenizer.decode(eval_dataset[i]["input_ids"]))
        print(tokenizer.decode([x for x in eval_dataset[i]["labels"] if x != -100]))
