from datasets import DatasetDict, load_dataset
from typing import Dict
import os
from data.base_builder import BaseBuilder
from data.svamp.env import SVAMPEnv

class SVAMPBuilder(BaseBuilder):

    def get_env_cls(self):
        return SVAMPEnv

    def _build_datasets(self) -> DatasetDict:
        data_name = self.config.get('data_name', 'official').strip()
        raw_dataset = load_dataset('ChilleD/SVAMP')
        raw_test_dataset = raw_dataset['test']
        if data_name == 'official':
            raw_train_dataset = raw_dataset['train']
        else:
            data_files = 'data/svamp/' + data_name + '.jsonl'
            if not os.path.exists(data_files):
                raise FileNotFoundError(f'Local svamp dataset not found: {data_files}')
            raw_train_dataset = load_dataset('json', data_files=data_files, split='train')
        val_size = int(len(raw_train_dataset) * self.config.get('val_ratio'))
        split = raw_train_dataset.train_test_split(test_size=val_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = (split['train'], split['test'])
        num_workers = 16
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
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
        format_template = 'Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \\boxed{}.'
        prompt_template = 'Question: {prompt}\n'
        question = str(example['question_concat']).strip()
        equation = str(example['Equation']).strip()
        answer = str(example['Answer']).strip()
        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = f'{equation}\n\\boxed{{{answer}}}'
        text_output = {'prompt': [{'role': 'user', 'content': processed_prompt}], 'completion': [{'role': 'assistant', 'content': processed_label}], 'solution': processed_label, 'test': processed_label}
        return text_output

    @classmethod
    def _keep_keys(cls):
        return ['prompt', 'completion', 'solution']
