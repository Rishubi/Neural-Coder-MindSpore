import os
from typing import *
from time import perf_counter
import math
import argparse
import ray
from ray.actor import ActorHandle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from human_eval.data import read_problems, write_jsonl

from utils import time_it, ProgressBar

from codex.data.preprocess import encode_whitespaces, decode_whitespaces


class CodeGenerator(object):
    def __init__(self, config: dict, model_ref, tokenizer_ref):
        self.device = torch.device(config['device'])
        self.problems: dict = read_problems()
        print('downloading model from ray ref ...')
        tic = perf_counter()
        model = ray.get(model_ref)
        toc = perf_counter()
        print(f'model downloaded, took {toc - tic} sec')
        torch.cuda.empty_cache()
        self.model = model.to(self.device)
        print('downloading tokenizer from ray ref ...')
        tic = perf_counter()
        tokenizer = ray.get(tokenizer_ref)
        toc = perf_counter()
        print(f'tokenizer downloaded, took {toc - tic} sec')
        self.tokenizer = tokenizer
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.min_length = config['min_length']
        self.max_length = config['max_length']
        self.num_samples = config['num_samples']

        self.whitespace_start_id = config['whitespace_start_id']
        self.max_whitespace_length = config['max_whitespace_length']

    def maybe_encode_whitespace(self, text: str):
        if self.max_whitespace_length > 0:
            result = encode_whitespaces(text, self.whitespace_start_id, self.max_whitespace_length)
            return result
        else:
            return text

    def maybe_decode_whitespace(self, text: str):
        if self.max_whitespace_length > 0:
            result = decode_whitespaces(text, self.whitespace_start_id, self.max_whitespace_length)
            return result
        else:
            return text

    def generate_text(self, prompt: str, max_length: int, num_return_sequences: int = 1) -> List[str]:
        prompt = self.maybe_encode_whitespace(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_length = input_ids.shape[1]
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=max_length,
            temperature=self.temperature,
            num_return_sequences=num_return_sequences,
            top_p=self.top_p,
        )
        gen_tokens = [s[input_length:] for s in gen_tokens]  # strip the input tokens
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        gen_text = [self.maybe_decode_whitespace(x) for x in gen_text]
        return gen_text

    @staticmethod
    def cleanup_text(text: str):
        lines = text.split('\n')
        cleaned_lines = []
        is_cleaned = False
        for line in lines:
            if len(line) > 0 and not line[0].isspace():
                is_cleaned = True
                break
            cleaned_lines.append(line)
        cleaned_text = '\n'.join(cleaned_lines)

        return cleaned_text, is_cleaned

    def generate_minibatch(self, task_id: str, minibatch_size: int):
        prompt = self.problems[task_id]['prompt']
        generated_texts = self.generate_text(prompt, max_length=self.max_length, num_return_sequences=minibatch_size)

        return [self.cleanup_text(x)[0] for x in generated_texts]

    def generate_batch(self, task_id: str, batch_size: int, minibatch_size: int, pba: ActorHandle):
        remaining = batch_size
        res = []

        while remaining > 0:
            step_size = min(minibatch_size, remaining)
            xs = self.generate_minibatch(task_id, step_size)
            pba.update.remote(step_size)
            res = res + xs

        return res

    def generate_one_sample(self, task_id: str):
        def recur(input_segs: List[str], current_max_len: int):
            # print(f'trying to generate {task_id} with max len {current_max_len}')
            prompt = ''.join(input_segs)
            generated_text = self.generate_text(prompt, max_length=current_max_len)[0]

            previously_generated = ''.join(input_segs[1:])
            all_generated = previously_generated + generated_text

            cleaned_text, touched = self.cleanup_text(all_generated)

            if not touched and current_max_len * 2 <= self.max_length:
                # print(f'generation not ended, trying 2x length')
                return recur(input_segs + [generated_text], current_max_len * 2)
            else:
                return cleaned_text

        prompt = self.problems[task_id]['prompt']
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        current_len = self.min_length
        while current_len <= prompt_len:
            current_len *= 2

        if current_len > self.max_length:
            raise RuntimeError(
                f'Maximal generation length is {self.max_length} while the prompt length is {prompt_len}')

        # print(f'generating one sample for {task_id}, prompt len {prompt_len}, initial seq len {current_len}')

        return recur([prompt], current_max_len=current_len)

    def generate_samples(self, task_id: str, num_samples: int, pba: ActorHandle):
        res = []
        for _ in range(num_samples):
            res.append(self.generate_one_sample(task_id))
            pba.update.remote(1)

        return res

    def generate_one_task(self, task_id: str, num_samples: int, pba: ActorHandle):
        results = self.generate_samples(task_id, num_samples, pba)
        return [{'task_id': task_id, 'completion': x} for x in results]


def split_work(task_ids: List[str], num_samples_per_task: int, num_workers: int):
    batch_size = 8
    batch_num_per_task = math.ceil(num_samples_per_task / batch_size)
    task_count = 0
    results = [[] for _ in range(num_workers)]
    for task_id in task_ids:
        for _ in range(batch_num_per_task):
            results[task_count % num_workers].append((task_id, batch_size))
            task_count += 1
    return results


@ray.remote(num_gpus=1, max_calls=1)
def perform_tasks(tasks, pba: ActorHandle):
    results = []
    total_tasks = len(tasks)
    gen = CodeGenerator(args.__dict__, model_ref, tokenizer_ref)
    for i, (task_id, num_samples) in enumerate(tasks):
        # print(f'generating task {i + 1} / {total_tasks} ...')
        tic = perf_counter()
        solutions = gen.generate_one_task(task_id, num_samples, pba)
        results += solutions
        toc = perf_counter()
        # print(f'generating task {i + 1} / {total_tasks} done! took {toc - tic:.4f} sec')

    return results


if __name__ == '__main__':
    ray.init('auto', runtime_env={'working_dir': '.', 'excludes': ['outputs/', 'wandb/']})
    print(ray.available_resources())

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_model', type=str, default='EleutherAI/gpt-j-6B')
    parser.add_argument('--load_state', type=str,
                        default='/share/yichen/chkpts/gpt-j-codeparrot-continued/checkpoint-13600.bin')
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=int, default=0.95)
    parser.add_argument('--min_length', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--save_path', type=str, default='gptj.jsonl')
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--whitespace_start_id', type=int, default=10)
    parser.add_argument('--max_whitespace_length', type=int, default=10)
    args = parser.parse_args()

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = math.floor(ray.available_resources()['GPU'])
    print(f'num workers = {num_workers}')
    problem_keys = list(read_problems().keys())
    # problem_keys = ['HumanEval/1', 'HumanEval/10']
    gen_tasks = split_work(problem_keys, num_samples_per_task=200, num_workers=num_workers)

    total_samples = len(problem_keys) * 200
    pb = ProgressBar(total_samples)
    pba = pb.actor

    with time_it('loading GPT model ...'):
        if args.load_state is not None:
            config = AutoConfig.from_pretrained(args.load_model)
            state_path = args.load_state
            print(f'loading state from {state_path} ...')
            model = AutoModelForCausalLM.from_pretrained(state_path, torch_dtype=torch.float16, config=config)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.load_model, torch_dtype=torch.float16)
    with time_it('uploading model ref ...'):
        model_ref = ray.put(model)
    with time_it('loading tokenizer ...'):
        tokenizer = AutoTokenizer.from_pretrained(args.load_model)
    with time_it('uploading tokenizer ref ...'):
        tokenizer_ref = ray.put(tokenizer)

    # args.device = 'cuda'
    # gen = CodeGenerator(args.__dict__, model_ref, tokenizer_ref)
    # res = gen.generate_one_task('HumanEval/50', num_samples=10)
    #
    # input('PRESS ANY KEY TO CONTINUE')

    results = [perform_tasks.remote(xs, pba) for xs in gen_tasks if len(xs) > 0]

    pb.print_until_done()

    results = ray.get(results)

    output_solutions = []
    # aggregate all solutions
    for res in results:
        output_solutions = output_solutions + res

    write_jsonl(args.save_path, output_solutions)
