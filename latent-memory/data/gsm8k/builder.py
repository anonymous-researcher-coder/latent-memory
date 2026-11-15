from datasets import DatasetDict, load_dataset
from typing import Dict
import os
from data.base_builder import BaseBuilder
from data.gsm8k.env import GSM8KEnv

class GSM8KBuilder(BaseBuilder):


    def get_env_cls(self):
        return GSM8KEnv

    def _build_datasets(self) -> DatasetDict:
        data_name = self.config.get('data_name').strip()
        raw_dataset = load_dataset('gsm8k', 'main')
        raw_test_dataset = raw_dataset['test']
        if data_name == 'official':
            raw_train_dataset = raw_dataset['train']
        else:
            data_files = 'data/gsm8k/' + data_name + '.jsonl'
            if not os.path.exists(data_files):
                raise FileNotFoundError(f'Local gsm8k dataset not found: {data_files}')
            raw_train_dataset = load_dataset('json', data_files=data_files, split='train')
        val_size = int(len(raw_train_dataset) * self.config.get('val_ratio'))
        split = raw_train_dataset.train_test_split(test_size=val_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = (split['train'], split['test'])
        num_workers = 16
        keep_keys = self._keep_keys()
        keep_keys_ext = keep_keys
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_workers).select_columns(keep_keys_ext)
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_workers).select_columns(keep_keys_ext)
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_workers).select_columns(keep_keys)
        dataset_dict = DatasetDict()
        dataset_dict['train'] = train_dataset
        dataset_dict['valid'] = valid_dataset
        dataset_dict['test'] = test_dataset
        return dataset_dict


    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()

    @classmethod
    def _preprocess(cls, example: Dict):

        def _preprocess_answer(answer: str) -> str:
            raw_answer_list = answer.split('\n####')
            rationale = raw_answer_list[0]
            clean_answer = raw_answer_list[-1].strip()
            boxed_answer = '\\boxed{' + clean_answer + '}'
            new_string = rationale + boxed_answer
            return new_string.strip()
        format_template = 'Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \\boxed{}.'
        prompt_template = 'Question: {prompt}\n'
        question = example['question'].strip()
        answer = example['answer'].strip()
        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = _preprocess_answer(answer)
        text_output = {'prompt': [{'role': 'user', 'content': processed_prompt}], 'completion': [{'role': 'assistant', 'content': processed_label}], 'solution': processed_label, 'test': processed_label}
        return text_output

    @classmethod
    def _keep_keys(cls):
        return ['prompt', 'completion', 'solution']
