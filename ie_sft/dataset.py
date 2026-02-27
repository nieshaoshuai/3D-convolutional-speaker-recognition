import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer


FIELD_KEYS = ["time", "location", "person", "organization", "title"]


@dataclass
class IESample:
    input: str
    output: str


def _build_answer(time: str, location: str, person: str, organization: str, title: str) -> str:
    return json.dumps(
        {
            "time": time,
            "location": location,
            "person": person,
            "organization": organization,
            "title": title,
        },
        ensure_ascii=False,
    )


def generate_sft_samples(num_samples: int = 100, seed: int = 42) -> List[Dict[str, str]]:
    random.seed(seed)

    persons = ["张伟", "王芳", "李娜", "刘洋", "陈晨", "赵磊", "杨雪", "周杰", "吴敏", "郑凯"]
    organizations = ["星海科技", "远航研究院", "晨光大学", "北方医院", "云图智能", "华城法院", "青禾传媒", "海蓝集团"]
    locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "苏州"]
    titles = ["发布会", "签约仪式", "技术论坛", "年度总结会", "招聘宣讲", "新品路演", "战略会议", "公益活动"]
    templates = [
        "{time}，{organization}在{location}举行{title}，负责人{person}出席并发言。",
        "据报道，{person}于{time}到{location}参加由{organization}组织的{title}。",
        "{organization}宣布：{time}将在{location}举办{title}，由{person}担任主讲。",
        "活动通知：{title}定于{time}在{location}举行，承办机构是{organization}，联系人{person}。",
    ]

    samples: List[Dict[str, str]] = []
    for idx in range(num_samples):
        year = random.choice(["2021年", "2022年", "2023年", "2024年", "2025年"])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        time = f"{year}{month}月{day}日"
        person = random.choice(persons)
        organization = random.choice(organizations)
        location = random.choice(locations)
        title = random.choice(titles)
        template = random.choice(templates)

        input_text = template.format(
            time=time,
            organization=organization,
            location=location,
            title=title,
            person=person,
        )
        output = _build_answer(time, location, person, organization, title)
        samples.append({"input": input_text, "output": output})

    return samples


def ensure_data_file(data_path: str, num_samples: int = 100, seed: int = 42) -> None:
    file_path = Path(data_path)
    if file_path.exists():
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_sft_samples(num_samples=num_samples, seed=seed)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


class IESFTDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> IESample:
        item = self.samples[idx]
        return IESample(input=item["input"], output=item["output"])


class DataCollatorForIESFT:
    def __init__(self, tokenizer: AutoTokenizer, max_source_len: int = 256, max_target_len: int = 128):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[IESample]) -> Dict[str, torch.Tensor]:
        inputs = [item.input for item in batch]
        targets = [item.output for item in batch]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        label_ids = labels["input_ids"]
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        model_inputs["raw_inputs"] = inputs
        model_inputs["raw_targets"] = targets
        return model_inputs


def load_samples(data_path: str) -> List[Dict[str, str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    model_name: str,
    data_path: str,
    train_ratio: float = 0.8,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    ensure_data_file(data_path=data_path, num_samples=100, seed=seed)
    samples = load_samples(data_path)

    dataset = IESFTDataset(samples)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForIESFT(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader, tokenizer
