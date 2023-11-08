import datasets

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("kiamesdavies/prometheus-grafana-dashboards-full-v3",
                                    split="train" if split == dataset_config.train_split else "test")

    prompt_template = (
        f"{B_INST} {B_SYS}Provided series of PromQL queries for several Grafana panels, generate a full Grafana dashboard.{E_SYS}```json\n{{designer_input}}\n```\n{E_INST}\n"
    )

    output_template = f"[RESULT]```json\n{{designer_output}}\n```[/RESULT]"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt_template.format(**sample),
            "output": output_template.format(**sample),
        }

    dataset = dataset.map(apply_prompt_template,
                          remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["output"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label,
                          remove_columns=list(dataset.features))

    return dataset