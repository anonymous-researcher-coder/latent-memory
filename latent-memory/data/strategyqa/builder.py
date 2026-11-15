import os
from typing import Dict
from datasets import DatasetDict, load_dataset
from data.base_builder import BaseBuilder
from data.strategyqa.env import StrategyQAEnv

class StrategyQABuilder(BaseBuilder):

    def get_env_cls(self):
        return StrategyQAEnv

    def _build_datasets(self) -> DatasetDict:
        data_name = self.config.get('data_name', 'official').strip()
        max_samples = int(self.config.get('max_samples', 0))
        raw_dataset = load_dataset('ChilleD/StrategyQA')
        raw_test_dataset = raw_dataset['test']
        if data_name == 'official':
            raw_train_full = raw_dataset['train']
        else:
            data_files = 'data/strategyqa/' + data_name + '.jsonl'
            if not os.path.exists(data_files):
                raise FileNotFoundError(f'Local strategyqa dataset not found: {data_files}')
            raw_train_full = load_dataset('json', data_files=data_files, split='train')
        val_ratio = float(self.config.get('val_ratio', self.config.get('valid_ratio', 0.1)))
        val_size = max(1, int(len(raw_train_full) * val_ratio))
        split = raw_train_full.train_test_split(test_size=val_size, shuffle=True, seed=42)
        raw_train_dataset, raw_valid_dataset = (split['train'], split['test'])
        if max_samples > 0 and max_samples < len(raw_train_dataset):
            raw_train_dataset = raw_train_dataset.select(range(max_samples))
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
    def _preprocess(cls, example: Dict) -> Dict:
        question = str(example['question']).strip()
        answer_bool = bool(example['answer'])
        answer_str = 'Yes' if answer_bool else 'No'
        facts = str(example.get('facts', '') or '').strip()
        if facts:
            completion = facts + f'\n\nTherefore, the final answer is \\boxed{{{answer_str}}}'
        else:
            completion = f'\\boxed{{{answer_str}}}'
        format_template = 'Answer the following yes/no question with proper reasoning, and make sure to put the FINAL ANSWER (Yes or No) inside \\boxed{}.'
        prompt_template = 'Question: {prompt}\n'
        processed_prompt = format_template + prompt_template.format(prompt=question)
        return {'prompt': [{'role': 'user', 'content': processed_prompt}], 'completion': [{'role': 'assistant', 'content': completion}], 'solution': completion}

    @classmethod
    def _keep_keys(cls):
        return ['prompt', 'completion', 'solution']
